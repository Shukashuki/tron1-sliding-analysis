"""Fresh output, explicit copy SHA and pure-source archive safety."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import package_ground_contact_v2_partial as mod


def test_existing_output_precedes_input_loading(tmp_path):
    with pytest.raises(ValueError,match="fresh output"):
        mod.run(SimpleNamespace(output_dir=tmp_path))


def test_copy_is_hash_bound_and_never_overwrites(tmp_path):
    source=tmp_path/"source.json";source.write_text('{"a":1}')
    with pytest.raises(ValueError,match="differs"):mod.copy_bound(source,tmp_path/"wrong.json","0"*64)
    target=tmp_path/"copied.json";result=mod.copy_bound(source,target,mod.sha256(source))
    assert target.read_bytes()==source.read_bytes() and result["sha256"]==mod.sha256(target)
    with pytest.raises(ValueError,match="fresh destination"):mod.copy_bound(source,target)


def test_source_archive_accepts_only_repo_pure_source(tmp_path):
    with pytest.raises(ValueError,match="pure code"):
        mod.source_archive(tmp_path,{"../outside.so":"0"*64})
    with pytest.raises(ValueError,match="source changed"):
        mod.source_archive(tmp_path,{"scripts/ground_contact_reward_v2.py":"0"*64})


def test_partial_status_and_fairness_claims_are_explicit():
    source=Path(mod.__file__).read_text()
    for literal in ("PARTIAL_EXPERIMENT_NOT_GOAL_COMPLETION","FORMAL_TRAINING_RUNNING_RESULTS_NOT_AVAILABLE",
                    "COMPLETED_RECORDINGS_CONTACT_CRITERION_FAILED","Not used for cross-controller numerical return comparison",
                    "PARENT_REPORTED_PASSED_NOT_FORMAL_PERFORMANCE"):
        assert literal in source
    assert '"goal_complete":False' in source
