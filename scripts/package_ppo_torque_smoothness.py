"""Package the isolated torque-rate trial as pure source/configuration only.

Packaging does not authorize or launch training. The separate dispatcher
requires the completed action-rate method to report needs_next_method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import package_ppo_smoothness as base


ROOT = Path(__file__).resolve().parents[1]
FILES = (*base.FILES,
    "scripts/ppo_leg_torque_smoothness.py",
    "scripts/train_ppo_torque_smoothness.py",
    "scripts/run_ppo_torque_smoothness.py",
    "scripts/package_ppo_torque_smoothness.py",
    "config/ppo_leg_torque_smoothness_v1.json",
    "tests/test_ppo_leg_torque_smoothness.py",
    "tests/test_train_ppo_torque_smoothness.py",
    "tests/test_run_ppo_torque_smoothness.py",
    "tests/test_package_ppo_torque_smoothness.py",
)


def package(output, root=ROOT):
    original = base.FILES
    try:
        base.FILES = FILES
        return base.package(output, root=root)
    finally:
        base.FILES = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(package(parser.parse_args().output), indent=2))
