from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import collect_ppo_cem_friction_failure_400hz as mod


def test_stable_status_has_original_bytes_hash(tmp_path):
    import hashlib
    path = tmp_path/"status.json"; path.write_text('{"status":"running","count":2}\n')
    value, digest = mod.stable_status(path)
    assert value == {"status": "running", "count": 2}
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_partial_json_is_not_promoted_to_a_partial_case(tmp_path):
    path = tmp_path/"status.json"; path.write_text('{"status":')
    with pytest.raises(RuntimeError, match="changed during snapshot"):
        mod.stable_status(path)


def test_fresh_manifest_guard_precedes_source_access(tmp_path, monkeypatch):
    path = tmp_path/"existing.json"; path.write_text("{}")
    monkeypatch.setattr(mod, "ppo_cases", lambda *a: pytest.fail("No source access before fresh guard"))
    with pytest.raises(ValueError, match="fresh manifest"):
        mod.run(NS(output_file=path))


def test_partial_counts_no_false_completion(tmp_path, monkeypatch):
    first = [{"status": "completed_verified"}]*34
    second = [{"status": "reused_completed_nominal"}]*2+[{"status": "pending"}]*30+[{"status": "running"}]*2
    monkeypatch.setattr(mod, "ppo_cases", lambda *a: (first, {"verified": True}))
    monkeypatch.setattr(mod, "cem_cases", lambda *a: (second, {"snapshot": True}))
    args = NS(output_file=tmp_path/"fresh.json", ppo_root=tmp_path, cem_root=tmp_path)
    result = mod.run(args)
    assert result["counts"] == {"expected": 68, "completed": 36, "pending_or_failed": 32}
    assert result["complete17x4_matrix"] is False
    assert result["snapshot_does_not_auto_update"] is True
    assert result["new_training"] is False and result["new_simulation"] is False
    assert json.loads(args.output_file.read_text()) == result


def test_optional_actual_CEM_live_snapshot_never_calls_pending_complete():
    root = mod.DEFAULT_ROOT/"cem"
    if not (root/"experiment.json").is_file():
        pytest.skip("Optional authorized local dispatcher")
    cases, proof = mod.cem_cases(root)
    assert len(cases) == 34 and {case["mu"] for case in cases} == set(mod.ppo.GRID)
    reused = [case for case in cases if case["reused_nominal"]]
    assert len(reused) == 2 and all(case["mu"] == .7 for case in reused)
    assert proof["completed_full_episode_count"] >= 2
    assert all(case["status"] != "smoke_completed" for case in cases)


def test_optional_actual_verified_PPO_17x2_retrieval():
    root = mod.DEFAULT_ROOT/"ppo"
    if not (root/"manifest.json").is_file():
        pytest.skip("Optional completed remoteCPU retrieval")
    cases, proof = mod.ppo_cases(root)
    assert len(cases) == 34 and all(case["status"] == "completed_verified" for case in cases)
    assert len(proof["file_ledger"]) == 162
    assert sum(case["reused_nominal"] for case in cases) == 2
