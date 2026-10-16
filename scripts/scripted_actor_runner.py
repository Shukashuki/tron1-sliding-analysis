"""Strict TorchScript-only SAC inference adapter for the existing RSL evaluator.

This is NOT an OnPolicyRunner or a PPO checkpoint converter. The checkpoint is
hashed, never unpickled; only the separately exported deterministic actor is
loaded. Adjacent manifests bind it to a completed SAC run (or an explicitly
permitted, clearly labeled smoke). Local hashes establish artifact consistency,
not authenticity of an untrusted downloaded model: use trusted run directories.

The frozen actor already includes input normalization. The outer normalizer is
Identity, matching Isaac Lab's exporter contract: deep-copy policy.actor and
apply the supplied normalizer exactly once. No Isaac/RSL import or simulator is
needed to import or unit-test this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _recorded_hash_matches(path, expected, label):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)):
        raise ValueError(f"Missing or invalid recorded {label} SHA256")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Recorded {label} SHA256 differs from the actual artifact")
    return actual


class _ScriptedPolicy(nn.Module):
    """Only the fields used by the pinned evaluator and Isaac Lab exporter."""

    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.actor = nn.Identity()
        self.actor_obs_normalizer = nn.Identity()
        # Pinned eval_tracking.py reads this at final report construction, after
        # all rollout/export checks. Only the existing 51-D policy group is used.
        self.obs_groups = {"policy": ["policy"]}

    def get_actor_obs(self, obs):
        # A TensorDict and a plain mapping both satisfy this exact interface.
        value = obs["policy"]
        if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[-1] != 51
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError("Expected finite raw float32 actor observations with shape (batch, 51)")
        return value

    def forward(self, obs):
        return self.actor(self.actor_obs_normalizer(self.get_actor_obs(obs)))


class ScriptedActorRunner:
    """Duck-compatible inference runner; optimization/training is unsupported."""

    def __init__(self, wrapped, runner_cfg, log_dir=None, device="cpu"):
        if not isinstance(runner_cfg, dict) or runner_cfg.get("algorithm") != "SAC":
            raise ValueError("ScriptedActorRunner requires an explicit SAC runner configuration")
        if not isinstance(runner_cfg.get("allow_smoke_actor", False), bool):
            raise ValueError("allow_smoke_actor must be an explicit boolean")
        if log_dir is not None:
            raise ValueError("ScriptedActorRunner supports inference only, not a training log directory")
        self.wrapped = wrapped
        self.runner_cfg = dict(runner_cfg)
        self.device = torch.device(device)
        self.alg = SimpleNamespace(policy=_ScriptedPolicy().to(self.device))
        self.loaded = False
        self.actor_file_path = None
        self.actor_sha256 = None
        self.checkpoint_sha256 = None
        self.inference_provenance = None

    def load(self, checkpoint, load_optimizer=False, map_location=None):
        if load_optimizer:
            raise ValueError("This inference adapter never loads optimizer or pickle checkpoints")
        self.loaded = False
        checkpoint = Path(checkpoint).resolve(strict=True)
        run_dir = checkpoint.parent
        manifest_path = run_dir / "manifest.json"
        status_path = run_dir / "run_status.json"
        contract_path = run_dir / "policy_contract.json"
        config_path = run_dir / "runner_config.json"
        manifest = _json_object(manifest_path)
        status = _json_object(status_path)
        contract = _json_object(contract_path)
        recorded_cfg = _json_object(config_path)
        if manifest.get("algorithm") != "SAC" or recorded_cfg.get("algorithm") != "SAC":
            raise ValueError("Source manifests must identify a SAC run, never a PPO checkpoint")
        if (not isinstance(self.runner_cfg.get("sac"), dict)
                or self.runner_cfg["sac"] != manifest.get("sac_config")
                or self.runner_cfg["sac"] != recorded_cfg.get("sac")):
            raise ValueError("Supplied SAC runner configuration differs from the source run")
        smoke = status.get("status") == "smoke_passed"
        if smoke:
            if not self.runner_cfg.get("allow_smoke_actor", False):
                raise ValueError("Smoke actor is not a trained policy; set allow_smoke_actor=True only for pipeline smoke")
            if (status.get("smoke_only") is not True or manifest.get("smoke_only") is not True
                    or status.get("trained_policy_claimed") is not False):
                raise ValueError("Inconsistent smoke-only SAC provenance")
        elif (status.get("status") != "training_completed" or status.get("smoke_only") is not False
                or manifest.get("smoke_only") is not False or status.get("trained_policy_claimed") is not True):
            raise ValueError("SAC inference requires completed final training or an explicitly allowed passed smoke")
        expected_checkpoint = "model_smoke.pt" if smoke else "model_final.pt"
        if checkpoint.name != expected_checkpoint:
            raise ValueError("SAC inference requires the source run's final checkpoint, not an intermediate or renamed model")
        if (contract != manifest.get("policy_contract") or contract != manifest.get("contract")
                or contract.get("actor_dim") != 51 or contract.get("action_dim") != 8
                or contract.get("policy_algorithm") != "SAC"
                or contract.get("actor_format") != "torchscript_raw_observation_deterministic_tanh"
                or self.runner_cfg["sac"].get("actor_dim") != 51
                or self.runner_cfg["sac"].get("action_dim") != 8):
            raise ValueError("Source policy contract must consistently specify raw 51-D SAC observations and 8 actions")
        height = manifest.get("height_regularization")
        height_hash = manifest.get("height_adapter_sha256")
        if (not isinstance(height, dict) or height.get("mode") != "vertical_acceleration_only"
                or height.get("absolute_height_reward") is not False
                or height.get("height_termination") is not False
                or height.get("reference_vertical_velocity_reward") is not False
                or contract.get("height_regularization") != height
                or not isinstance(height_hash, str) or len(height_hash) != 64
                or any(char not in "0123456789abcdef" for char in height_hash)):
            raise ValueError("SAC actor lacks consistent height-smoothness training provenance")
        task_settings = manifest.get("task_profile", {}).get("height_regularization")
        if (not isinstance(task_settings, dict)
                or any(height.get(key) != task_settings.get(key)
                       for key in ("mode", "weight", "acceleration_scale_m_s2"))
                or any(key not in task_settings for key in ("mode", "weight", "acceleration_scale_m_s2"))):
            raise ValueError("Height-smoothness task profile differs from the trained policy contract")
        actor_path = (run_dir / ("actor_smoke.pt" if smoke else "actor_normalized.pt")).resolve(strict=True)
        if actor_path.parent != run_dir:
            raise ValueError("The exported actor must reside in the source run directory")
        checkpoint_hash = _recorded_hash_matches(checkpoint, status.get("checkpoint_sha256"), "checkpoint")
        actor_hash = _recorded_hash_matches(actor_path, status.get("actor_sha256"), "actor")
        location = torch.device(self.device if map_location is None else map_location)
        actor = torch.jit.load(str(actor_path), map_location=location).eval()
        with torch.inference_mode():
            probe = torch.stack((torch.zeros(51, device=location), torch.linspace(-1, 1, 51, device=location)))
            first, repeated = actor(probe), actor(probe)
            if (not isinstance(first, torch.Tensor) or first.shape != (2, 8)
                    or not torch.isfinite(first).all() or not (first.abs() <= 1).all()
                    or not torch.equal(first, repeated)):
                raise ValueError("Exported SAC actor must produce deterministic finite normalized actions with shape (batch, 8)")
        self.device = location
        self.alg.policy.actor = actor
        self.alg.policy.to(self.device).eval()
        self.actor_file_path = str(actor_path)
        self.actor_sha256 = actor_hash
        self.checkpoint_sha256 = checkpoint_hash
        self.inference_provenance = {
            "algorithm": "SAC", "actor_format": contract["actor_format"],
            "actor_file": self.actor_file_path, "actor_sha256": actor_hash,
            "checkpoint_file": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": sha256(manifest_path), "run_status_sha256": sha256(status_path),
            "policy_contract_sha256": sha256(contract_path), "runner_config_sha256": sha256(config_path),
            "height_adapter_sha256": height_hash, "height_regularization": height,
            "smoke_only": smoke, "trained_policy_claimed": not smoke,
            "outer_normalizer": "identity; normalization is embedded in the exported actor",
            "checkpoint_deserialized": False,
        }
        self.loaded = True
        return self.inference_provenance

    def get_inference_policy(self, device=None):
        if not self.loaded:
            raise RuntimeError("Load and verify the SAC actor before requesting inference")
        if device is not None:
            self.device = torch.device(device)
            self.alg.policy.to(self.device).eval()

        @torch.inference_mode()
        def infer(obs):
            if not self.loaded:
                raise RuntimeError("SAC actor is not currently verified")
            raw = self.alg.policy.get_actor_obs(obs)
            if raw.device != self.device:
                raise ValueError("SAC observations and actor must be on the requested inference device")
            action = self.alg.policy.actor(raw)
            if (action.shape != (raw.shape[0], 8) or not torch.isfinite(action).all()
                    or not (action.abs() <= 1).all()):
                raise FloatingPointError("SAC actor produced invalid normalized actions")
            return action

        return infer
