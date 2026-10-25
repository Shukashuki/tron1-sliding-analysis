"""Guard against accepting an old K actor or a wrong validation geometry."""
from pathlib import Path
import hashlib
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import eval_ppo_k_native_slices9 as evaluation


def test_parent_keeps_dependency_root_but_binds_new_cohort_and_wrapper_identity():
    source = ROOT/"scripts/eval_ppo_ground_contact_v2k.py"
    before = source.read_bytes()
    parent = evaluation.load_parent(ROOT)
    assert hashlib.sha256(before).hexdigest() == evaluation.verification.EVALUATOR_SHA
    assert parent.ROOT == ROOT
    assert parent.COHORT_ID == evaluation.training_model.COHORT_ID
    assert parent.provenance.__globals__["__file__"] == str(Path(evaluation.__file__).resolve())
    assert parent.prepare_plan.__globals__["verify_training"] is parent.verify_training
    assert parent.validate_output.__globals__["verify_training"] is parent.verify_training
    assert source.read_bytes() == before


def test_old_training_protocol_cannot_enter_new_model_evaluator(tmp_path):
    parent = evaluation.load_parent(ROOT)
    with pytest.raises(ValueError, match="Wrong any-wheel contact cohort"):
        parent.verify_training(tmp_path, ROOT/"config/ppo_ground_contact_v2k_400hz_v1.json")


def test_incomplete_new_model_training_is_rejected(tmp_path):
    parent = evaluation.load_parent(ROOT)
    with pytest.raises(ValueError, match="Missing or indirect completed training artifact"):
        parent.verify_training(tmp_path, ROOT/"config/ppo_k_native_slices9_400hz_v1.json")
