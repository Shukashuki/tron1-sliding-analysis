"""Fake-child queue tests: no SSH, training, actor inference or simulator jobs."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


REAL_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("matrix400", REAL_ROOT/"scripts/run_ppo_backend_400hz_matrix.py")
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class Guard:
    @staticmethod
    def write_json(path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch):
    root = tmp_path/"snapshot"; (root/"scripts").mkdir(parents=True); (root/"config").mkdir()
    monkeypatch.setattr(matrix, "ROOT", root)
    protocol = root/"config/ppo_backend_400hz_comparison_v1.json"
    protocol.write_bytes((REAL_ROOT/"config/ppo_backend_400hz_comparison_v1.json").read_bytes())
    driver = root/"scripts/run_ppo_backend_400hz_pair.py"; driver.write_text("# frozen fake dispatcher\n")
    monkeypatch.setattr(matrix, "TRAINING_DRIVER_SHA256", matrix.sha256(driver))
    for name in ("analyze_ppo_backend_400hz_matrix.py", "guard.py", "probe.py"):
        (root/"scripts"/name).write_text("# pure fixture source\n")
    legacy, bm, training, preflight = [tmp_path/n for n in ("legacy", "bm", "training", "preflight")]
    for path in (legacy, bm, training, preflight):
        path.mkdir()
    inputs = {}
    for name in ("model", "motion", "asset"):
        path = tmp_path/(name+".input"); path.write_bytes(("frozen-"+name).encode()); inputs[name] = path
    args = NS(training_pair_dir=training, training_dispatcher_pid=None, preflight_dir=preflight,
        source_snapshot=root, legacy_source=legacy, model=inputs["model"], motion=inputs["motion"],
        asset_path=inputs["asset"], beyondmimic_source=bm, python=Path("/explicit/venv/bin/python"),
        gpu_probe_source=root/"scripts/probe.py", guard_source=root/"scripts/guard.py",
        output_dir=tmp_path/"matrix", physical_gpu_index=4, protocol=None)
    identity = {"uuid": "gpu4uuid", "pci_bus_id": "0000:04:00.0"}
    pair = {"stage": "formal", "pid": 1234, "physical_gpu_index": 4,
        "status": "training_pair_completed_validation_pending", "initial_policy_state_sha256": "a"*64,
        "dispatcher_argv": [str(driver), "--preflight-dir", str(preflight)], "selected_gpu_identity": identity,
        "children": [{"stage": label, "status": "completed_verified", "returncode": 0} for label in ("physx_formal", "mjlab_formal")]}
    Guard.write_json(training/"backend_400hz_pair_status.json", pair)
    Guard.write_json(preflight/"backend_400hz_pair_status.json", {"status": "preflight_completed_requires_formal_authorization",
        "physical_gpu_index": 4, "children": [{"stage": label, "status": "completed_verified", "returncode": 0}
        for label in ("parity", "physx_smoke", "mjlab_smoke")]})
    cohorts = {}
    for phase, parent, suffix in (("pipeline_smoke", preflight, "smoke"), ("formal", training, "formal")):
        for origin in matrix.ORIGINS:
            directory = parent/(origin+"_"+suffix); directory.mkdir()
            paths = {}
            for name in ("model_final.pt", "policy.pt", "manifest.json", "runner_config.json", "cohort_completion_audit.json"):
                path = directory/name; path.write_text("frozen fixture actor metadata")
                paths[name] = path
            cohorts[directory] = {"directory": directory, "paths": paths,
                "input_sha256": {name: matrix.sha256(path) for name, path in paths.items()},
                "training_backend": origin, "smoke_only": phase == "pipeline_smoke",
                "manifest": {"initial_policy_state_sha256": "a"*64}}
    preflight_status = json.loads((preflight/"backend_400hz_pair_status.json").read_text())
    for phase in ("pipeline_smoke", "formal"):
        for index, origin in enumerate(matrix.ORIGINS):
            parent, suffix = (preflight, "smoke") if phase == "pipeline_smoke" else (training, "formal")
            cohort = cohorts[parent/(origin+"_"+suffix)]
            proof = {"status": "verified", "backend": origin, "run_kind": suffix,
                "training_environment_steps": 1536 if phase == "pipeline_smoke" else 12288000,
                "checkpoint_sha256": cohort["input_sha256"]["model_final.pt"],
                "exported_policy_sha256": cohort["input_sha256"]["policy.pt"],
                "manifest_sha256": cohort["input_sha256"]["manifest.json"],
                "cohort_completion_audit_sha256": cohort["input_sha256"]["cohort_completion_audit.json"],
                "initial_policy_state_sha256": "a"*64}
            (preflight_status["children"][index+1] if phase == "pipeline_smoke" else pair["children"][index])["verification"] = proof
    Guard.write_json(training/"backend_400hz_pair_status.json", pair)
    Guard.write_json(preflight/"backend_400hz_pair_status.json", preflight_status)
    calls, validations, gpu_calls = [], [], []

    class Shared:
        INPUT_PINS = {name: matrix.sha256(path) for name, path in inputs.items()}
        @staticmethod
        def verify_protocol(path):
            return json.loads(Path(path).read_text())
        @staticmethod
        def verify_cohort(directory, protocol_path, *, allow_smoke=False):
            value = cohorts[Path(directory)]
            assert allow_smoke is value["smoke_only"]
            return value
        @staticmethod
        def validate_output(folder, cohort_dir, protocol_path, *, validation_backend, allow_smoke=False):
            value = cohorts[Path(cohort_dir)]
            assert allow_smoke is value["smoke_only"]
            validations.append((Path(folder), validation_backend, allow_smoke))
            return {"status": "verified", "native_states": 2793, "policy_decisions": 349, "duration_s": 6.98,
                "physics_dt_s": .0025, "policy_dt_s": .02, "validation_backend": validation_backend,
                "training_backend": value["training_backend"], "smoke_only": allow_smoke,
                "recording_sha256": "b"*64, "report_sha256": "c"*64}

    def gpu(index):
        gpu_calls.append(index)
        return {"idle": True, "physical_index": index, "pci_bus_id_cuda_ordinal_verified": True,
                "all_reported_compute_and_graphics_processes": [], **identity}

    def child(command, log, cwd, environment, spawned):
        calls.append((command, environment.copy()))
        assert str(cwd) == str(root)
        spawned(2000+len(calls))
        output = Path(command[command.index("--output-dir")+1]); output.mkdir(parents=True)
        if "analyze_ppo_backend_400hz_matrix.py" in command[2]:
            (output/"summary.json").write_text("{}")
            assert environment["CUDA_VISIBLE_DEVICES"] == ""
        elif "_isaac.py" in command[2]:
            assert "CUDA_VISIBLE_DEVICES" not in environment
            assert "--physical-gpu-index" in command and command[command.index("--physical-gpu-index")+1] == "4"
        else:
            assert environment["CUDA_VISIBLE_DEVICES"] == ""
        return 0
    return NS(args=args, pair=pair, cohorts=cohorts, shared=Shared, guard=Guard, gpu=gpu, child=child,
              calls=calls, validations=validations, gpu_calls=gpu_calls)


def run_fixture(f, **overrides):
    return matrix.run(f.args, shared=f.shared, guard=f.guard, gpu_probe=overrides.get("gpu", f.gpu),
        child_runner=overrides.get("child", f.child), sleep=overrides.get("sleep", lambda _: None),
        process_check=lambda *_: {"process_signalled": False})


def test_complete_pipeline_then_68_cells_nominal_first_and_cpu_analysis(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    status = run_fixture(f)
    assert status["status"] == "completed_with_analysis"
    assert status["counts"] == {"pipeline_smoke_expected": 4, "pipeline_smoke_completed": 4,
        "formal_expected": 68, "formal_completed": 68, "runtime_failed": 0}
    assert len(f.calls) == 73 and len(f.validations) == 72 and len(f.gpu_calls) == 36
    assert all(c["smoke_only"] and c["mu"] == .7 for c in status["cases"][:4])
    assert all(not c["smoke_only"] and c["mu"] == .7 for c in status["cases"][4:8])
    assert all(c["status"] == "completed_verified" for c in status["cases"])
    assert "--allow-smoke-policy" in f.calls[0][0] and "--allow-smoke-policy" not in f.calls[4][0]
    manifest = json.loads((f.args.output_dir/"validation_matrix_manifest.json").read_text())
    assert manifest["status"] == "validation_matrix_completed" and manifest["protocol_path"]
    assert manifest["cases"][0]["verification"]["native_states"] == 2793


def test_wait_for_running_training_allocates_no_gpu_and_does_not_freeze_actors(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    running = copy.deepcopy(f.pair); running["status"] = "running_physx_formal"
    Guard.write_json(f.args.training_pair_dir/"backend_400hz_pair_status.json", running)
    observations = []
    def sleep(seconds):
        if seconds == 30.:
            assert not f.calls and not f.gpu_calls
            static = json.loads((f.args.output_dir/"static_frozen_source_and_inputs.json").read_text())
            assert static["formal_actor_bytes_frozen"] is False
            assert not any(p.endswith("model_final.pt") for p in static["files"])
            observations.append(seconds)
            Guard.write_json(f.args.training_pair_dir/"backend_400hz_pair_status.json", f.pair)
    result = run_fixture(f, sleep=sleep)
    assert result["status"] == "completed_with_analysis" and observations == [30.]


@pytest.mark.parametrize("state", ["stopped_without_retry", "failed", "training_pair_completed_but_invalid"])
def test_failed_training_pair_never_submits_a_validation_job(tmp_path, monkeypatch, state):
    f = fixture(tmp_path, monkeypatch)
    f.pair["status"] = state; Guard.write_json(f.args.training_pair_dir/"backend_400hz_pair_status.json", f.pair)
    with pytest.raises(ValueError, match="failed/stopped"):
        run_fixture(f)
    assert not f.calls and not f.gpu_calls


def test_busy_gpu_aborts_without_cpu_fallback_or_retry(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    def busy(index):
        row = f.gpu(index); row["idle"] = False; row["all_reported_compute_and_graphics_processes"] = [{"pid": "foreign"}]
        return row
    with pytest.raises(ValueError, match="busy"):
        run_fixture(f, gpu=busy)
    assert not f.calls and len(f.gpu_calls) == 1


def test_failed_owned_child_is_not_retried(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    def fail(*args):
        f.child(*args); return 1
    with pytest.raises(ValueError, match="child failed"):
        run_fixture(f, child=fail)
    assert len(f.calls) == 1
    status = json.loads((f.args.output_dir/"backend_400hz_matrix_status.json").read_text())
    assert status["counts"]["runtime_failed"] == 1 and status["counts"]["formal_completed"] == 0


def test_source_change_after_a_returned_child_stops_before_next_job(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    def change(*args):
        code = f.child(*args)
        (f.args.source_snapshot/"scripts/analyze_ppo_backend_400hz_matrix.py").write_text("changed")
        return code
    with pytest.raises(ValueError, match="Frozen source"):
        run_fixture(f, child=change)
    assert len(f.calls) == 1


def test_completed_actor_change_is_detected(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    def change(*args):
        code = f.child(*args)
        next(iter(f.cohorts.values()))["paths"]["policy.pt"].write_text("changed actor")
        return code
    with pytest.raises(ValueError, match="Frozen source"):
        run_fixture(f, child=change)
    assert len(f.calls) == 1


def test_inconsistent_initial_network_prevents_all_jobs(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    next(iter(f.cohorts.values()))["manifest"]["initial_policy_state_sha256"] = "d"*64
    with pytest.raises(ValueError, match="initialization"):
        run_fixture(f)
    assert not f.calls


def test_parent_training_artifact_proof_must_match_actual_completed_cohort(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f.pair["children"][0]["verification"]["checkpoint_sha256"] = "x"*64
    Guard.write_json(f.args.training_pair_dir/"backend_400hz_pair_status.json", f.pair)
    with pytest.raises(ValueError, match="Parent training/preflight child proof"):
        run_fixture(f)
    assert not f.calls


def test_incomplete_native_completion_proof_is_rejected(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    original = f.shared.validate_output
    def invalid(*args, **kwargs):
        proof = original(*args, **kwargs); proof["native_states"] = 1397; return proof
    monkeypatch.setattr(f.shared, "validate_output", invalid)
    with pytest.raises(ValueError, match="completion proof"):
        run_fixture(f)
    assert len(f.calls) == 1


def test_never_adopt_changed_sources_while_freezing_newly_completed_actors(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    original = f.shared.verify_cohort
    def change(*args, **kwargs):
        (f.args.source_snapshot/"scripts/analyze_ppo_backend_400hz_matrix.py").write_text("new unapproved source")
        return original(*args, **kwargs)
    monkeypatch.setattr(f.shared, "verify_cohort", change)
    with pytest.raises(ValueError, match="Frozen source"):
        run_fixture(f)
    assert not f.calls and not f.gpu_calls


def test_never_adopt_actor_changes_between_verification_and_freeze(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    original = f.shared.verify_cohort
    def change(*args, **kwargs):
        value = original(*args, **kwargs)
        value["paths"]["policy.pt"].write_text("unverified changed actor")
        return value
    monkeypatch.setattr(f.shared, "verify_cohort", change)
    with pytest.raises(ValueError, match="changed while freezing"):
        run_fixture(f)
    assert not f.calls


def test_full_matrix_is_preserved_if_cpu_analysis_fails(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    def fail_analysis(*args):
        code = f.child(*args)
        return 1 if "analyze_ppo_backend_400hz_matrix.py" in args[0][2] else code
    with pytest.raises(ValueError, match="Analysis failed"):
        run_fixture(f, child=fail_analysis)
    status = json.loads((f.args.output_dir/"backend_400hz_matrix_status.json").read_text())
    assert status["status"] == "analysis_failed_validation_preserved"
    assert status["counts"]["formal_completed"] == 68 and status["counts"]["runtime_failed"] == 0
    assert (f.args.output_dir/"validation_matrix_manifest.json").is_file()
