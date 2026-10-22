"""Ensure isolated beta-off identity while preserving frozen plant execution."""
import ast
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ppo_backend_400hz_beta_contract as contract
import eval_ppo_backend_400hz_beta_off_isaac as physx
import eval_ppo_backend_400hz_beta_off_mujoco as native


@pytest.mark.parametrize("wrapper", [physx, native])
def test_all_imported_contract_functions_are_rebound_and_wrapper_identity_is_new(wrapper):
    module = wrapper.evaluator()
    for name in wrapper.CONTRACT_EXPORTS:
        assert getattr(module, name) is getattr(contract, name)
    assert Path(module.__file__).resolve() == Path(wrapper.__file__).resolve()
    assert module.parser().get_default("cohort_protocol") == contract.DEFAULT_PROTOCOL
    assert module.verify_cohort is not __import__("ppo_backend_400hz_contract").verify_cohort


@pytest.mark.parametrize("wrapper,name", [(physx, "eval_tracking_friction_isaac.py"),
                                          (native, "eval_tracking_mujoco.py")])
def test_unchanged_counted_plant_source_and_no_file_mutation(wrapper, name):
    root = ROOT / "outputs/controller_evaluation_source_20261015_v2"
    path = root / "scripts" / name
    if not path.is_file():
        pytest.skip("Externally provisioned frozen evaluator is unavailable")
    module = wrapper.evaluator()
    before = contract.sha256(path)
    result = module.materialize_source(path.read_text(), root)
    ast.parse(result)
    assert contract.sha256(path) == before == module.BASE_SHA256
    with pytest.raises(ValueError, match="source changed|content differs"):
        module.materialize_source(path.read_text() + "\n", root)
