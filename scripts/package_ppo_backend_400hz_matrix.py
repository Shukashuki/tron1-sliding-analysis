"""Package a separate immutable validation/analysis source tree, never weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import package_ppo_backend_400hz as training_packet


ROOT = Path(__file__).resolve().parents[1]
EXTRA_FILES = (
    "scripts/package_ppo_backend_400hz_matrix.py",
    "scripts/run_ppo_backend_400hz_matrix.py",
    "scripts/ppo_backend_400hz_contract.py",
    "scripts/eval_ppo_backend_400hz_isaac.py",
    "scripts/eval_ppo_backend_400hz_mujoco.py",
    "scripts/analyze_ppo_backend_400hz_matrix.py",
    "scripts/analyze_ppo_smoothness.py",
    "scripts/summarize_ppo_smoothness.py",
    "scripts/analyze_ppo_contact_stability.py",
    "scripts/analyze_ppo_contact_timestep.py",
    "scripts/analyze_ppo_ground_support_failure.py",
    "scripts/controller_metrics.py",
    "scripts/controller_trace.py",
    "tests/test_package_ppo_backend_400hz_matrix.py",
    "tests/test_run_ppo_backend_400hz_matrix.py",
    "tests/test_ppo_backend_400hz_contract.py",
    "tests/test_eval_ppo_backend_400hz.py",
    "tests/test_analyze_ppo_backend_400hz_matrix.py",
)


def build_file_list(root=ROOT):
    root = Path(root).resolve()
    files = (*training_packet.build_file_list(root), *EXTRA_FILES)
    if len(files) != len(set(files)):
        raise ValueError("Duplicate validation source entry")
    for name in files:
        path = root/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Missing regular allowlisted validation source: " + name)
    return files


def package(output, root=ROOT):
    helper = training_packet._load_delegate()
    helper.FILES = build_file_list(root)

    def render(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("kind") == "ppo_smoothness_pure_source_snapshot":
            value = {**value, "kind": "ppo_backend_400hz_validation_matrix_source_snapshot",
                "contents": "explicit source/config/tests and existing license notices; no assets, weights, motion or credentials",
                "training_snapshot_modified": False,
                "purpose": "wait for completed training, four pipeline smoke rollouts, then68 formal episodes and offline analysis",
                "upstream_mjlab_commit": training_packet.MJLAB_PIN,
                "legacy_evaluation_source": "existing external immutable source; verified before execution"}
        return json.dumps(value, *args, **kwargs)

    helper.json = SimpleNamespace(dumps=render)
    return helper.package(output, root=root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(package(parser.parse_args().output), indent=2))
