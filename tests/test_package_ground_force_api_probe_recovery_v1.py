"""Publication safety and exact offline replay of failed-finalization arrays."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import package_ground_force_api_probe_recovery_v1 as mod


def test_no_overwrite_before_input_read(tmp_path):
    with pytest.raises(ValueError, match="Fresh public"):
        mod.run(SimpleNamespace(output_dir=tmp_path, input_root=tmp_path/"missing"))


def test_actual_failure_excerpt_retains_error_but_not_absolute_remote_paths():
    text = 'warning\nTraceback (most recent call last):\n  File "/secret/remote/probe.py", line216, in finish\n    write_json(plan["input_dir"]/"evaluation_source_audit.json", audit)\nFileExistsError: File exists: \'/secret/remote/evaluation_source_audit.json\'\n'
    result = mod.failure_excerpt(text)
    assert "FileExistsError" in result["excerpt"] and "evaluation_source_audit.json" in result["excerpt"]
    assert "/secret/remote" not in result["excerpt"] and "warning" not in result["excerpt"]
    with pytest.raises(ValueError): mod.failure_excerpt("FileExistsError without actual traceback")


def test_source_capture_contains_frozen_v1_r2_and_recovery_without_models():
    ledger = mod.source_ledger()
    assert ledger["scripts/eval_ppo_ground_force_api_probe_v2.py"] == mod.recovery.V1_SHA
    assert "scripts/eval_ppo_ground_force_api_probe_v2r2.py" in ledger
    assert "scripts/recover_ground_force_api_probe_v1.py" in ledger
    assert all(Path(name).suffix in (".py", ".json") for name in ledger)


def test_claims_keep_failure_and_unexecuted_r2_explicit():
    source = Path(mod.__file__).read_text()
    assert '"finalization_completed": False' in source and '"original_A_formal_result": False' in source
    assert '"r2_fix_execution_claimed": False' in source
    assert '"new_physics_or_retry_executed_for_recovery": False' in source
