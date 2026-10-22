"""Beta-off normalized PPO in PhysX; unchanged native 400 Hz evaluator."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import ppo_backend_400hz_beta_contract as contract

BASE_SHA256 = "e670d6c88bd85eb8aa36f402301e7f0c9ae5f7d3b97336a78d1a66750a642f99"
CONTRACT_EXPORTS = ("DEFAULT_PROTOCOL", "StrictPPOExportRunner", "prepare_inputs", "provenance", "require",
                    "sha256", "verify_cohort", "verify_legacy", "verify_recording", "write_json")


def evaluator():
    path = Path(__file__).with_name("eval_ppo_backend_400hz_isaac.py")
    contract.require(hashlib.sha256(path.read_bytes()).hexdigest() == BASE_SHA256,
                     "Frozen PhysX 400 Hz evaluator wrapper source differs")
    spec = importlib.util.spec_from_file_location("_beta_off_physx_evaluator", path)
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
