"""Fresh admission without modifying native action/recording/reset lifecycle."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import eval_contact_taskspace_scratch as evaluation
import continue_contact_taskspace_scratch as continuation


def test_fresh_provenance_does_not_inherit_warm_identity_or_optimizer():
    source = (ROOT/"scripts/eval_contact_warm_refinement.py").read_text()
    result = evaluation.materialize(source)
    assert "training.WARM_STATE_SHA" not in result and "training.CHECKPOINT_SHA" not in result
    assert "source_K_actor_sha256" not in result and "additional_refinement_transitions" not in result
    assert "from_scratch=True,historical_checkpoint_used=False" in result
    assert "initial_policy_state_sha256=training.RANDOM_STATE_SHA" in result
    assert "_scratch_config=lambda config:config" in result
    assert "import verify_contact_taskspace_scratch as verification" in result
    assert "scratch_contact_evaluation_status.json" in result
    assert "verification.verify_training(d,p,allow_smoke=allow_smoke)" in result
    # Native inference and full-horizon lifecycle are left to the same parent.
    assert "parent._finish(report,arrays,current)" in result
    assert "_new_configure=physx.configure" in result
    assert "parent.common.StrictPPOExportRunner(cohort,device)" in result
    assert "plan['mu']==.7" in result


def test_changed_frozen_parent_is_rejected():
    source = (ROOT/"scripts/eval_contact_warm_refinement.py").read_text()
    with pytest.raises(AssertionError): evaluation.materialize(source+'\n')


def test_continuation_waits_for_both_exact_fresh_budgets_without_timeout():
    source = (ROOT/"scripts/continue_contact_warm_refinement.py").read_text()
    result = continuation.materialize(source)
    assert "proof['transitions']==12288000" in result
    assert "2457600" not in result and "14745600" not in result
    assert "assert proof['from_scratch'] and not proof['historical_checkpoint_used']" in result
    assert "deadline" not in result
    assert "p['start_ticks']==launch['start_ticks'] and p['argv']==launch['argv']" in result
    assert "row['returncode']=p.returncode;assert p.returncode==0" in result
    assert "helper.query_gpus()" in result and "if g['idle']" in result
    assert "source_K_" not in result
    assert "scratch_contact_evaluation_status.json" in result
    with pytest.raises(AssertionError): continuation.materialize(source+'\n')
