"""Explicit, immutable beta-off source snapshot, excluding all trained assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import package_ppo_backend_400hz_matrix as baseline_packet


ROOT = Path(__file__).resolve().parents[1]
EXTRA_FILES = (
    "scripts/train_ppo_backend_400hz_beta_off.py",
    "scripts/ppo_backend_400hz_beta_contract.py",
    "scripts/run_ppo_backend_400hz_beta_off.py",
    "scripts/eval_ppo_backend_400hz_beta_off_isaac.py",
    "scripts/eval_ppo_backend_400hz_beta_off_mujoco.py",
    "scripts/package_ppo_backend_400hz_beta_off.py",
    "config/ppo_backend_400hz_beta_off_v1.json",
    "tests/test_ppo_backend_400hz_beta_off.py",
    "tests/test_run_ppo_backend_400hz_beta_off.py",
    "tests/test_eval_ppo_backend_400hz_beta_off.py",
    "tests/test_package_ppo_backend_400hz_beta_off.py",
)


def build_file_list(root=ROOT):
    root = Path(root).resolve()
    files = (*baseline_packet.build_file_list(root), *EXTRA_FILES)
    if len(files) != len(set(files)):
        raise ValueError("Duplicate beta-off source entry")
    for name in files:
        path = root / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Missing regular allowlisted beta-off source: " + name)
    return files


def package(output, root=ROOT):
    helper = baseline_packet.training_packet._load_delegate()
    helper.FILES = build_file_list(root)
    def render(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("kind") == "ppo_smoothness_pure_source_snapshot":
            value = {**value, "kind": "ppo_backend_400hz_beta_off_source_snapshot",
                "contents": "explicit source/config/tests and existing license notices; no assets, weights, motion or credentials",
                "baseline_source_files_edited": False,
                "purpose": "isolated beta-off PhysX/MJWarp smoke, formal training and four nominal validation episodes",
                "intervention": "reward_weights.sideslip:2.0->0.0 only",
                "upstream_mjlab_commit": baseline_packet.training_packet.MJLAB_PIN}
        return json.dumps(value, *args, **kwargs)
    helper.json = SimpleNamespace(dumps=render)
    return helper.package(output, root=root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(package(parser.parse_args().output), indent=2))
