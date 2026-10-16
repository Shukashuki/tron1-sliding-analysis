"""Small, simulator-independent SAC core for the TRON1 comparison.

Algorithm provenance (independent implementation, not copied library code):
Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, equations
5--10, Section 5 (entropy dual), and Appendix C (squashed densities):
https://arxiv.org/html/1812.05905v2 . Twin minimum-Q targets and target-network
Polyak updates also follow the official modern-SAC description:
https://spinningup.openai.com/en/latest/algorithms/sac.html . The log-temperature
loss below uses the usual log-alpha surrogate with the same entropy fixed point.
Time-limit handling follows the environment API's distinction between termination
and truncation: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/ .

The actor consumes raw 51-D observations; privileged 120-D observations are used
ONLY by the critics. Dimensions are configurable. All 8 actions are normalized
to [-1, 1]; the environment, not this core, applies the shared actuator mapping.
Raw replay observations undergo explicitly fixed affine normalization inside
the networks (identity by default). No implicit/online normalization or reward
rescaling is performed. The exported deterministic actor includes normalization.

Callers MUST supply bootstrap observations from the actual successor state,
including pre-reset final observations on truncation. This module cannot recover
them from an auto-reset environment. True termination masks bootstrapping;
truncation alone does not. Discounts are supplied per transition, e.g. gamma for
one control interval or gamma**n for an n-step transition whose reward was already
accumulated correctly. No training, simulator integration, or parity claim is
implied by this implementation.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SACConfig:
    actor_dim: int = 51
    critic_dim: int = 120
    action_dim: int = 8
    hidden_sizes: tuple[int, ...] = (256, 128, 128)
    activation: str = "elu"
    learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    initial_alpha: float = 0.2
    target_entropy: Optional[float] = None
    tau: float = 0.005
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    max_grad_norm: float = 10.0

    def __post_init__(self):
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1
               for x in (self.actor_dim, self.critic_dim, self.action_dim, *self.hidden_sizes)):
            raise ValueError("Network dimensions must be positive integers")
        if not self.hidden_sizes:
            raise ValueError("At least one hidden layer is required")
        if self.activation not in ("elu", "relu"):
            raise ValueError("SAC activation must be explicitly 'elu' or 'relu'")
        for name in ("learning_rate", "alpha_learning_rate", "initial_alpha", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid positive SAC setting: {name}")
        if not math.isfinite(self.tau) or not 0 < self.tau <= 1:
            raise ValueError("tau must lie in (0, 1]")
        if (not math.isfinite(self.log_std_min) or not math.isfinite(self.log_std_max)
                or not -20 <= self.log_std_min < self.log_std_max <= 2):
            raise ValueError("Require -20 <= log_std_min < log_std_max <= 2")
        if self.target_entropy is not None and not math.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite")


class FixedNormalizer(nn.Module):
    """Explicit (raw - mean) / scale; never changes during replay or export."""

    def __init__(self, dim, mean=None, scale=None):
        super().__init__()
        mean = torch.zeros(dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        scale = torch.ones(dim) if scale is None else torch.as_tensor(scale, dtype=torch.float32)
        if (mean.shape != (dim,) or scale.shape != (dim,) or not torch.isfinite(mean).all()
                or not torch.isfinite(scale).all() or not (scale > 0).all()):
            raise ValueError("Normalization requires finite vectors and positive scales")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self.mean) / self.scale


def _mlp(input_dim, hidden_sizes, activation):
    layers = []
    activation_type = nn.ELU if activation == "elu" else nn.ReLU
    for width in hidden_sizes:
        layers.extend((nn.Linear(input_dim, width), activation_type()))
        input_dim = width
    return nn.Sequential(*layers)


class SquashedActor(nn.Module):
    def __init__(self, config, mean=None, scale=None):
        super().__init__()
        self.normalizer = FixedNormalizer(config.actor_dim, mean, scale)
        self.trunk = _mlp(config.actor_dim, config.hidden_sizes, config.activation)
        self.mu = nn.Linear(config.hidden_sizes[-1], config.action_dim)
        self.log_std = nn.Linear(config.hidden_sizes[-1], config.action_dim)
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """Deterministic evaluation action, not the mean of the squashed density."""
        return torch.tanh(self.mu(self.trunk(self.normalizer(raw))))

    def sample(self, raw):
        hidden = self.trunk(self.normalizer(raw))
        mu = self.mu(hidden)
        log_std = self.log_std(hidden).clamp(self.log_std_min, self.log_std_max)
        normal = torch.distributions.Normal(mu, log_std.exp())
        pre_tanh = normal.rsample()
        # log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u));
        # avoid log(1-a*a), which becomes -inf at floating-point saturation.
        correction = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        log_prob = (normal.log_prob(pre_tanh) - correction).sum(-1)
        return torch.tanh(pre_tanh), log_prob


class TwinQ(nn.Module):
    def __init__(self, config, mean=None, scale=None):
        super().__init__()
        self.normalizer = FixedNormalizer(config.critic_dim, mean, scale)
        self.q1 = nn.Sequential(_mlp(config.critic_dim + config.action_dim, config.hidden_sizes, config.activation),
                                nn.Linear(config.hidden_sizes[-1], 1))
        self.q2 = nn.Sequential(_mlp(config.critic_dim + config.action_dim, config.hidden_sizes, config.activation),
                                nn.Linear(config.hidden_sizes[-1], 1))

    def forward(self, raw, action):
        value = torch.cat((self.normalizer(raw), action), dim=-1)
        return self.q1(value).squeeze(-1), self.q2(value).squeeze(-1)


class VectorReplay:
    """Bounded device-resident ring buffer; capacity counts transitions, not steps.

    add() requires all successor observations explicitly, even on episode ends.
    On truncation those must be final/pre-reset observations, NOT reset outputs.
    This semantic requirement cannot be established from tensor values alone.
    """

    def __init__(self, capacity, actor_dim=51, critic_dim=120, action_dim=8, device="cpu"):
        dims = (capacity, actor_dim, critic_dim, action_dim)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in dims):
            raise ValueError("Replay capacity and dimensions must be positive integers")
        self.capacity = capacity
        self.actor_dim, self.critic_dim, self.action_dim = actor_dim, critic_dim, action_dim
        self.device = torch.device(device)
        widths = {"actor_obs": actor_dim, "critic_obs": critic_dim, "actions": action_dim,
                  "bootstrap_actor_obs": actor_dim, "bootstrap_critic_obs": critic_dim,
                  "rewards": None, "discounts": None, "terminated": None, "truncated": None}
        self.storage = {name: torch.empty((capacity,) if width is None else (capacity, width),
                        dtype=torch.bool if name in ("terminated", "truncated") else torch.float32,
                        device=self.device) for name, width in widths.items()}
        self.position = self.size = self.total_added = 0

    @property
    def allocated_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.storage.values())

    @torch.no_grad()
    def add(self, *, actor_obs, critic_obs, actions, rewards, bootstrap_actor_obs,
            bootstrap_critic_obs, terminated, truncated, discounts):
        values = dict(actor_obs=actor_obs, critic_obs=critic_obs, actions=actions, rewards=rewards,
                      bootstrap_actor_obs=bootstrap_actor_obs, bootstrap_critic_obs=bootstrap_critic_obs,
                      terminated=terminated, truncated=truncated, discounts=discounts)
        for key, value in values.items():
            value = torch.as_tensor(value, device=self.device).detach()
            if key in ("terminated", "truncated") and value.dtype != torch.bool:
                raise ValueError("terminated and truncated must be explicit boolean vectors")
            values[key] = value.to(dtype=self.storage[key].dtype)
        first = values["actor_obs"]
        if first.ndim != 2 or len(first) == 0:
            raise ValueError("Replay insertion requires a nonempty batch")
        count = len(first)
        for key, value in values.items():
            expected = (count, *self.storage[key].shape[1:])
            if value.shape != expected or not torch.isfinite(value).all():
                raise ValueError(f"Invalid replay field shape or nonfinite value: {key}")
        if not ((values["discounts"] >= 0) & (values["discounts"] <= 1)).all():
            raise ValueError("Per-transition discounts must lie in [0, 1]")
        if not (values["actions"].abs() <= 1.0).all():
            raise ValueError("Replay actions must be normalized to [-1, 1]")
        # Write only the most recent capacity transitions: repeated advanced
        # indices are ambiguous on CUDA when a vector batch exceeds capacity.
        retained = min(count, self.capacity)
        indexes = (self.position + count - retained + torch.arange(retained, device=self.device)) % self.capacity
        for key, value in values.items():
            self.storage[key][indexes] = value[-retained:]
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.total_added += count

    def sample(self, batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1 or self.size == 0:
            raise ValueError("Sampling requires nonempty replay and positive batch size")
        indexes = torch.randint(self.size, (batch_size,), device=self.device)
        return {name: value[indexes] for name, value in self.storage.items()}

    def state_dict(self):
        return {"capacity": self.capacity, "actor_dim": self.actor_dim, "critic_dim": self.critic_dim,
                "action_dim": self.action_dim, "position": self.position, "size": self.size,
                "total_added": self.total_added,
                "storage": {name: value[:self.size].detach().cpu().clone()
                            for name, value in self.storage.items()}}

    @torch.no_grad()
    def load_state_dict(self, state):
        if any(state[name] != getattr(self, name) for name in
               ("capacity", "actor_dim", "critic_dim", "action_dim")):
            raise ValueError("Replay checkpoint shape/capacity mismatch")
        size, position, total = state["size"], state["position"], state["total_added"]
        if (not 0 <= size <= self.capacity or not 0 <= position < self.capacity
                or total < size or size != min(self.capacity, total) or position != total % self.capacity):
            raise ValueError("Invalid replay checkpoint counters")
        if set(state["storage"]) != set(self.storage):
            raise ValueError("Replay checkpoint field mismatch")
        for name, value in state["storage"].items():
            if (value.shape != self.storage[name][:size].shape or value.dtype != self.storage[name].dtype
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid replay checkpoint tensor: {name}")
        for name, value in state["storage"].items():
            self.storage[name][:size].copy_(value.to(self.device))
        self.size, self.position, self.total_added = size, position, total


@torch.no_grad()
def soft_q_target(rewards, discounts, terminated, truncated, next_q, next_log_prob, alpha):
    """Truncation bootstraps; termination (including both flags true) does not."""
    shape = rewards.shape
    if (rewards.ndim != 1 or any(value.shape != shape for value in
            (discounts, terminated, truncated, next_q, next_log_prob))
            or terminated.dtype != torch.bool or truncated.dtype != torch.bool):
        raise ValueError("SAC target fields must be equally shaped vectors with boolean flags")
    continuation = rewards + discounts * (next_q - alpha * next_log_prob)
    return torch.where(terminated, rewards, continuation)


def entropy_temperature_loss(log_alpha, log_prob, target_entropy):
    """Temperature update must not backpropagate into the policy's density."""
    return -(log_alpha * (log_prob.detach() + target_entropy)).mean()


class SACAgent:
    """Asymmetric SAC: deployable actor observations, privileged training critics."""

    def __init__(self, config=None, device="cpu", *, actor_mean=None, actor_scale=None,
                 critic_mean=None, critic_scale=None):
        self.config = config or SACConfig()
        self.device = torch.device(device)
        self.actor = SquashedActor(self.config, actor_mean, actor_scale).to(self.device)
        self.critic = TwinQ(self.config, critic_mean, critic_scale).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(self.config.initial_alpha), device=self.device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_learning_rate)
        self.target_entropy = (-float(self.config.action_dim) if self.config.target_entropy is None
                               else self.config.target_entropy)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, raw_actor_obs, deterministic=False):
        raw = torch.as_tensor(raw_actor_obs, dtype=torch.float32, device=self.device)
        if raw.ndim not in (1, 2) or raw.shape[-1] != self.config.actor_dim or not torch.isfinite(raw).all():
            raise ValueError("Invalid raw actor observation")
        return self.actor(raw) if deterministic else self.actor.sample(raw)[0]

    def _step(self, loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite SAC loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        return float(norm.detach())

    def update(self, batch):
        """One gradient update; batch normally comes from VectorReplay.sample()."""
        required = {"actor_obs", "critic_obs", "actions", "rewards", "bootstrap_actor_obs",
                    "bootstrap_critic_obs", "terminated", "truncated", "discounts"}
        if set(batch) != required:
            raise ValueError("SAC update requires all explicit transition fields")
        batch = {name: value.to(self.device).detach() for name, value in batch.items()}
        count = len(batch["rewards"])
        widths = {"actor_obs": self.config.actor_dim, "critic_obs": self.config.critic_dim,
                  "bootstrap_actor_obs": self.config.actor_dim,
                  "bootstrap_critic_obs": self.config.critic_dim, "actions": self.config.action_dim}
        for name, value in batch.items():
            expected = (count, widths[name]) if name in widths else (count,)
            if count == 0 or value.shape != expected or not torch.isfinite(value).all():
                raise ValueError(f"Invalid SAC batch field: {name}")
        if not ((batch["discounts"] >= 0) & (batch["discounts"] <= 1)).all():
            raise ValueError("Per-transition discounts must lie in [0, 1]")
        if not (batch["actions"].abs() <= 1).all():
            raise ValueError("SAC batch actions must lie in [-1, 1]")
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch["bootstrap_actor_obs"])
            next_q1, next_q2 = self.target_critic(batch["bootstrap_critic_obs"], next_action)
            target = soft_q_target(batch["rewards"], batch["discounts"], batch["terminated"],
                                   batch["truncated"], torch.minimum(next_q1, next_q2), next_log_prob,
                                   self.alpha.detach())
        q1, q2 = self.critic(batch["critic_obs"], batch["actions"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_norm = self._step(critic_loss, self.critic_optimizer, self.critic.parameters())
        self.critic_optimizer.zero_grad(set_to_none=True)
        # Freeze Q parameters, but differentiate Q(s, a_theta) through its action.
        self.critic.requires_grad_(False)
        try:
            action, log_prob = self.actor.sample(batch["actor_obs"])
            q1_pi, q2_pi = self.critic(batch["critic_obs"], action)
            actor_loss = (self.alpha.detach() * log_prob - torch.minimum(q1_pi, q2_pi)).mean()
            actor_norm = self._step(actor_loss, self.actor_optimizer, self.actor.parameters())
        finally:
            self.critic.requires_grad_(True)
        alpha_loss = entropy_temperature_loss(self.log_alpha, log_prob, self.target_entropy)
        self._step(alpha_loss, self.alpha_optimizer, [self.log_alpha])
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.lerp_(param, self.config.tau)
            # Normalizers are fixed, but copy buffers explicitly for checkpoint parity.
            for target_buffer, buffer in zip(self.target_critic.buffers(), self.critic.buffers()):
                target_buffer.copy_(buffer)
        self.updates += 1
        metrics = {"critic_loss": critic_loss, "actor_loss": actor_loss, "alpha_loss": alpha_loss,
                   "alpha": self.alpha, "entropy": -log_prob.mean(), "target_mean": target.mean()}
        result = {name: float(value.detach()) for name, value in metrics.items()}
        result.update(critic_grad_norm=critic_norm, actor_grad_norm=actor_norm, updates=self.updates)
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("Nonfinite SAC metrics")
        return result

    def state_dict(self):
        """Includes all optimization state; save() optionally adds replay and RNG."""
        return {"schema_version": 1, "config": asdict(self.config), "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(), "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().clone(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(), "updates": self.updates,
                "normalization": "fixed_affine_in_network_identity_by_default"}

    def load_state_dict(self, state):
        if state.get("schema_version") != 1 or SACConfig(**state["config"]) != self.config:
            raise ValueError("SAC checkpoint configuration mismatch")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.updates = state["updates"]

    def save(self, path, replay=None):
        """Trusted torch checkpoint; replay is optional and may be large."""
        state = {"agent": self.state_dict(), "torch_rng": torch.get_rng_state(),
                 # A CPU save must not initialize contexts on unrelated GPUs.
                 "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                 "replay": None if replay is None else replay.state_dict()}
        torch.save(state, Path(path))

    def load(self, path, replay=None, restore_rng=True):
        """Load only trusted locally generated checkpoints (uses torch pickle)."""
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if replay is not None and state.get("replay") is None:
            raise ValueError("Checkpoint does not contain resumable replay")
        self.load_state_dict(state["agent"])
        if replay is not None:
            replay.load_state_dict(state["replay"])
        if restore_rng:
            torch.set_rng_state(state["torch_rng"])
            if (state["cuda_rng"] is not None) != (self.device.type == "cuda"):
                raise ValueError("CPU/CUDA changed; pass restore_rng=False for nonidentical resume")
            if state["cuda_rng"] is not None:
                torch.cuda.set_rng_state(state["cuda_rng"], self.device)

    def export_actor(self, path):
        """TorchScript deterministic raw-observation actor, with fixed normalization."""
        actor = copy.deepcopy(self.actor).cpu().eval()
        torch.jit.script(actor).save(str(path))
