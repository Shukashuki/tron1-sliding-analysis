"""Beta-off normalized PPO in native CPU MuJoCo; unchanged 400 Hz evaluator."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import ppo_backend_400hz_beta_contract as contract

BASE_SHA256 = "9887aa0b4b7ef74845f47f8623356490a9990e4449e0b42f4a77b2b35a49c1e4"
CONTRACT_EXPORTS = ("DEFAULT_PROTOCOL", "prepare_inputs", "provenance", "require", "sha256", "verify_cohort",
                    "verify_legacy", "verify_recording", "write_json")


def evaluator():
    path = Path(__file__).with_name("eval_ppo_backend_400hz_mujoco.py")
    contract.require(hashlib.sha256(path.read_bytes()).hexdigest() == BASE_SHA256,
                     "Frozen native 400 Hz evaluator wrapper source differs")
    spec = importlib.util.spec_from_file_location("_beta_off_native_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in CONTRACT_EXPORTS:
        setattr(module, name, getattr(contract, name))
    module.__file__ = __file__
    return module


def run(argv):
    return evaluator().run(argv)


if __name__ == "__main__":
    run(sys.argv[1:])
