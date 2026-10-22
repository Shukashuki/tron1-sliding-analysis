from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_ppo_friction_failure_400hz as mod


def test_full17grid_two_actors_and_exact_nominal_reuse():
    paths = mod.layout("/data/zengxiong/workspaces/tron1-ppo-beta-ablation-400hz-20261022")
    cases = mod.cases(Path("/tmp/NEW_VALIDATION"), paths)
    assert len(cases) == 34 and {case["mu"] for case in cases} == set(mod.GRID)
    assert len({case["case_id"] for case in cases}) == 34
    assert sum(case["reused_nominal"] for case in cases) == 2
    assert [case["mu"] for case in cases[:8]] == [.7, .7, .15, .15, .3, .3, 1.2, 1.2]
    for case in cases:
        assert case["checkpoint_sha256"] == mod.CHECKPOINTS[case["condition"]]
        assert case["validation_backend"] == "native CPU MuJoCo"


def test_fresh_guard_before_interpreter_environment_or_actor_access(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "source_ledger", lambda *a: pytest.fail("No source/actor access before fresh guard"))
    with pytest.raises(ValueError, match="fresh output"):
        mod.run(NS(output_dir=tmp_path, workspace=tmp_path))


def test_current_source_pins_are_literal_frozen_local_bytes():
    assert all(mod.sha256(ROOT/name) == digest for name, digest in mod.SOURCE_PINS.items())
    assert all(len(value) == 64 for value in mod.CHECKPOINTS.values())


def fake_args(tmp_path, monkeypatch, fail_one=False):
    paths = {"snapshot": tmp_path/"snapshot", "cohorts": {"beta_on": tmp_path/"on_cohort", "beta_off": tmp_path/"off_cohort"},
        "nominal": {"beta_on": tmp_path/"on_nominal", "beta_off": tmp_path/"off_nominal"},
        "motion": tmp_path/"motion.npz", "model": tmp_path/"model.xml", "legacy": tmp_path/"eval_tracking_mujoco.py"}
    for path in (paths["snapshot"], *paths["cohorts"].values()):
        path.mkdir()
    monkeypatch.setattr(mod, "layout", lambda *a: paths)
    monkeypatch.setattr(mod, "source_ledger", lambda *a: dict(mod.SOURCE_PINS))
    for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "" if name == "CUDA_VISIBLE_DEVICES" else "1")
    calls, verified = [], []
    def module(label):
        def cohort(*args):
            return {"training_backend": "physx", "smoke_only": False,
                "input_sha256": {"model_final.pt": mod.CHECKPOINTS[label], "policy.pt": mod.ACTORS[label]}}
        def validate(folder, *args, **kwargs):
            if folder == paths["nominal"][label]:
                mu, pins = .7, mod.NOMINAL_PINS[label]
            else:
                mu = float(folder.parent.name[2:].replace("p", "."))
                pins = {"recording_sha256": "a"*64, "report_sha256": "b"*64}
            verified.append((label, mu))
            return {"status": "verified", "checkpoint_sha256": mod.CHECKPOINTS[label], "evaluation_contact_friction": mu,
                "native_states": 2793, "policy_decisions": 349, **pins}
        return NS(verify_cohort=cohort, validate_output=validate)
    modules = {"ppo_backend_400hz_contract": module("beta_on"), "ppo_backend_400hz_beta_contract": module("beta_off")}
    monkeypatch.setattr(mod.importlib, "import_module", lambda name: modules[name])
    def child(command, **kwargs):
        calls.append(command)
        mu = float(command[command.index("--evaluation-contact-friction")+1])
        return NS(returncode=1 if fail_one and len(calls) == 1 else 0)
    monkeypatch.setattr(mod.subprocess, "run", child)
    return NS(output_dir=tmp_path/"fresh_evaluations", workspace=tmp_path, python=Path(sys.executable), driver_sha256="c"*64), calls, verified


def test_mock_complete32_new_CPU_jobs_and_two_nominal_reuses(tmp_path, monkeypatch):
    args, calls, verified = fake_args(tmp_path, monkeypatch)
    old_path = list(sys.path)
    try:
        result = mod.run(args)
    finally:
        sys.path[:] = old_path
    assert result["status"] == "completed" and len(calls) == 32 and len(verified) == 34
    assert result["counts"] == {"expected": 34, "completed": 34, "failed": 0, "reused_nominal": 2}
    assert result["new_GPU_job"] is False and result["original_training_or_queue_modified"] is False
    assert all(float(call[call.index("--evaluation-contact-friction")+1]) != .7 for call in calls)
    assert json.loads((args.output_dir/"status.json").read_text())["status"] == "completed"


def test_failed_case_not_retried_or_promoted_to_success(tmp_path, monkeypatch):
    args, calls, verified = fake_args(tmp_path, monkeypatch, fail_one=True)
    old_path = list(sys.path)
    try:
        result = mod.run(args)
    finally:
        sys.path[:] = old_path
    assert result["status"] == "completed_with_failures"
    assert len(calls) == 32 and len(verified) == 33 and result["counts"]["failed"] == 1
    assert result["cases"][2]["status"] == "failed"


def test_cuda_exposure_rejected_before_source_or_cohort(tmp_path, monkeypatch):
    args, _, _ = fake_args(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setattr(mod, "source_ledger", lambda *a: pytest.fail("GPU exposed; reject before reading actors"))
    with pytest.raises(ValueError, match="hidden CUDA"):
        mod.run(args)


def test_no_silent_NaN_in_status(tmp_path):
    with pytest.raises(ValueError):
        mod.write_status(tmp_path/"bad.json", {"value": float("nan")})
