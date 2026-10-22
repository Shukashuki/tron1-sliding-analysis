from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_ppo_backend_400hz_nominal as mod
from test_analyze_ppo_backend_400hz_matrix import signal_fixture


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def refresh(root, manifest):
    manifest["files"] = [{"path": str(p.relative_to(root)), "size": p.stat().st_size, "sha256": mod.sha256(p)}
        for p in sorted(root.rglob("*")) if p.is_file() and p != root / "manifest.json"]
    dump(root / "manifest.json", manifest)


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "payload"
    root.mkdir()
    reference, model, legacy = (tmp_path / name for name in ("reference.npz", "model.xml", "legacy.py"))
    for path in (reference, model, legacy):
        path.write_bytes(path.name.encode())
    protocol = mod.verify_protocol()
    protocol_path = mod.DEFAULT_PROTOCOL
    protocol_sha = mod.sha256(protocol_path)
    (root / "config").mkdir()
    (root / "config/ppo_backend_400hz_comparison_v1.json").write_bytes(protocol_path.read_bytes())
    profile = {"action_rate_weight": -.5, "reward_weights": {"grounded": .4},
        "height_regularization": {"mode": "vertical_acceleration_only"}, "contact_friction": 1.}
    metadata, original, cases, by_origin = {}, [], [], {}
    for oi, origin in enumerate(mod.ORIGINS):
        checkpoint, actor = str(oi + 1) * 64, str(oi + 3) * 64
        common = {"run_kind": "formal", "smoke_only": False, "trained_policy_claimed": True,
                  "initial_policy_state_sha256": "a" * 64}
        contract = {**common, "contract_status": "trained_completed_normalized_actor", "policy_algorithm": "PPO",
            "policy_training_backend": origin, "checkpoint_sha256": checkpoint, "exported_policy_sha256": actor,
            "physics_dt": .0025, "decimation": 8, "policy_fps": 50., "actor_dim": 51, "critic_dim": 120,
            "action_dim": 8, "leg_kp": 500., "leg_kd": 10., "actor_includes_observation_normalizer": True,
            "runtime_clock_and_interface_verified": True,
            "physics_clock_origin": "trained_at_native400Hz_not_evaluation_only_refinement",
            "reference_file_sha256": mod.INPUT_PINS["motion"], "training_cohort_protocol_sha256": protocol_sha,
            "trained_task_profile": profile, "task_profile": profile}
        status = {**common, "status": "training_completed", "backend": origin, "training_environment_steps": 12288000,
            "checkpoint_sha256": checkpoint, "exported_policy_sha256": actor}
        startup = {**common, "runtime": {"status": "verified"}}
        training_manifest = {**common, "backend": origin, "cohort": common}
        training_manifest.pop("run_kind")  # Real top-level manifest has no run_kind.
        files = {"final_policy_contract": ("final_policy_contract.json", contract), "run_status": ("run_status.json", status),
            "startup_audit": ("cohort_startup_audit.json", startup), "manifest": ("manifest.json", training_manifest)}
        metadata[origin] = {}
        hashes = {}
        for role, (name, value) in files.items():
            relative = f"training/{origin}/{name}"
            dump(root / relative, value)
            metadata[origin][role] = relative
            hashes[name] = mod.sha256(root / relative)
        completion = {**common, "status": "completed_verified", "backend": origin, "cohort_id": protocol["cohort_id"],
            "training_environment_steps": 12288000, "seed": 42, "protocol_sha256": protocol_sha,
            "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02, "action_rate_weight": -.5,
            "grounded_reward_weight": .4, "source_files_unchanged": True, "additional_reward_terms": False,
            "historical_checkpoint_used": False, "checkpoint_sha256": checkpoint, "exported_policy_sha256": actor,
            "final_policy_contract_sha256": hashes["final_policy_contract.json"],
            "manifest_sha256": hashes["manifest.json"], "startup_audit_sha256": hashes["cohort_startup_audit.json"]}
        relative = f"training/{origin}/cohort_completion_audit.json"
        dump(root / relative, completion)
        metadata[origin]["completion_audit"] = relative
        hashes["cohort_completion_audit.json"] = mod.sha256(root / relative)
        by_origin[origin] = (checkpoint, actor, hashes)
    for origin in mod.ORIGINS:
        checkpoint, actor, hashes = by_origin[origin]
        for engine, backend in mod.analysis.ENGINE_LABELS.items():
            directory = f"cases/{origin}/{engine}"
            folder = root / directory
            folder.mkdir(parents=True)
            raw_file = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
            np.savez(folder / raw_file, fixture=np.array([1.]))
            audit = {"status": "verified", "training_backend": origin, "validation_backend": backend,
                "run_kind": "formal", "smoke_only": False, "trained_policy_claimed": True,
                "checkpoint_sha256": checkpoint, "normalized_actor_sha256": actor, "protocol_sha256": protocol_sha,
                "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02, "native_states": 2793,
                "policy_decisions": 349, "duration_s": 6.98, "old_200hz_actor_used": False,
                "additional_training_updates": 0, "normalizer_included_in_actor": True, "outer_normalizer": "Identity",
                "cohort_input_sha256": hashes,
                "wrapper_sha256": mod.sha256(ROOT / "scripts" / ("eval_ppo_backend_400hz_isaac.py" if engine == "physx"
                                                                  else "eval_ppo_backend_400hz_mujoco.py"))}
            deployment = {"physics_dt": .0025, "decimation": 8, "task_profile": {**profile, "contact_friction": .7},
                "trained_task_profile": profile}
            report = {"backend_400hz_cohort_evaluation": audit, "contact_material_audit": {}}
            if engine == "physx":
                report.update(status="evaluated", deterministic_start_frame=0, domain_randomization_enabled=False,
                    observation_noise_enabled=False, policy_export=deployment,
                    physics_trajectory_sha256=mod.sha256(folder / raw_file))
            else:
                report.update(status="completed", completed_full_reference=True, hidden_resets=0, root_state_writes=1,
                    start_reference_frame=0, domain_parameters_resampled_during_episode=False,
                    policy_sha256=actor, contract=deployment)
            dump(folder / "report.json", report)
            proof = {"status": "verified", "native_states": 2793, "policy_decisions": 349, "duration_s": 6.98,
                "physics_dt_s": .0025, "policy_dt_s": .02, "validation_backend": backend, "training_backend": origin,
                "smoke_only": False, "evaluation_contact_friction": .7, "checkpoint_sha256": checkpoint,
                "recording_sha256": mod.sha256(folder / raw_file), "report_sha256": mod.sha256(folder / "report.json")}
            case = {"case_id": origin + "_to_" + engine, "training_backend": origin, "engine_tag": engine,
                "validation_backend": backend, "mu": .7, "directory": directory,
                "remote_directory": f"/remote/matrix/formal/{origin}/{engine}/mu0p700", "raw_file": raw_file,
                "report_file": "report.json", "recording_sha256": proof["recording_sha256"],
                "report_sha256": proof["report_sha256"], "checkpoint_sha256": checkpoint,
                "actor_sha256": actor, "verification": proof}
            cases.append(case)
            original.append({**case, "phase": "formal", "status": "completed_verified", "directory": case["remote_directory"]})
    matrix_status = {"status": "running_formal_other_case", "counts": {"formal_expected": 68, "formal_completed": 46,
        "pipeline_smoke_expected": 4, "pipeline_smoke_completed": 4, "runtime_failed": 0}}
    dump(root / "matrix_status_snapshot.json", {**matrix_status, "cases": original})
    manifest = {"schema_version": 1, "kind": mod.KIND, "cohort_id": protocol["cohort_id"], "nominal_mu": .7,
        "nominal_only": True, "full_matrix_complete": False, "observed_utc": "2026-10-22T10:00:00Z",
        "matrix_status": matrix_status, "reference_sha256": mod.sha256(reference), "model_sha256": mod.sha256(model),
        "cases": cases, "training_metadata": metadata}
    refresh(root, manifest)
    trace, ref_values, signals = signal_fixture(.03)
    metrics = {"windows": {name: mod.analysis.window_metrics(trace, ref_values, signals, interval) for name, interval in
        (("core", (1.44, 4.)), ("full", (0., 6.98)), ("reference_overlap", (0., 6.96)))},
        "xy_screen": {"large_xy_error": False}, "wheel_heading_FK_audit": {"dynamics_stepping": False}}
    monkeypatch.setattr(mod, "verify_recording", lambda p, b: {key: cases[0]["verification"][key] for key in
        ("status", "native_states", "policy_decisions", "duration_s", "physics_dt_s", "policy_dt_s")} |
        {"recording_sha256": mod.sha256(p)})
    monkeypatch.setattr(mod.analysis, "reference_data", lambda p: ref_values)
    monkeypatch.setattr(mod.analysis, "load_fk", lambda *a: (object(), {"isolated_FK_only": True}))
    monkeypatch.setattr(mod.analysis, "analyze_case", lambda *a: (copy.deepcopy(trace), copy.deepcopy(metrics)))
    monkeypatch.setattr(mod.analysis, "material_summary", lambda *a: {})
    args = NS(input_root=root, reference=reference, model=model, legacy_mujoco_evaluator=legacy,
              cohort_protocol=protocol_path, output_dir=tmp_path / "output")
    return args, manifest


def test_four_nominal_records_verified_without_weights_and_partial_label(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path, monkeypatch)
    traces, ref, summary = mod.load_nominal(args.input_root, args.reference, args.model,
                                          args.legacy_mujoco_evaluator, args.cohort_protocol)
    assert set(traces) == {(o, e) for o in mod.ORIGINS for e in mod.analysis.ENGINE_LABELS}
    assert len(summary["rows"]) == 4 and summary["nominal_cells_completed"] == 4
    assert summary["matrix_status"]["counts"]["formal_completed"] == 46
    assert summary["full_matrix_complete"] is False and summary["nominal_only"] is True
    assert summary["checkpoint_or_actor_downloaded_or_loaded"] is False
    assert not list(args.input_root.rglob("*.pt"))


def test_outputs_explicit_partial_and_four_csv_rows_and_shared_plot_bounds(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch)
    summary = mod.run(args)
    overview = json.loads((args.output_dir / "overview.json").read_text())
    assert overview["full_matrix_complete"] is False and len(overview["nominal_four_cells"]) == 4
    assert "tested_mu_lists" not in overview
    assert len((args.output_dir / "metric_table.csv").read_text().splitlines()) == 5
    assert (args.output_dir / "nominal_2x2_topview.png").is_file()
    settings = summary["plot_settings"]
    assert "PARTIAL (46/68 observed)" in settings["title"]
    assert settings["shared_world_bounds"]["xlim"][0] == pytest.approx(-.35)
    assert settings["contact_gates_arrow_opacity"] is False and "spin invariant" in settings["wheel_arrow_kind"]


@pytest.mark.parametrize("change,match", [
    (lambda m: m.update(full_matrix_complete=True), "nominal-only"),
    (lambda m: m["cases"].__setitem__(1, m["cases"][0]), "unique nominal"),
    (lambda m: m["cases"][0].update(actor_sha256="f" * 64), "actor/checkpoint"),
    (lambda m: m["cases"][0]["verification"].update(native_states=1397), "original queue"),
    (lambda m: m["matrix_status"]["counts"].update(formal_completed=68), "original queue snapshot"),
])
def test_reject_mislabelled_mixed_or_modified_proofs(tmp_path, monkeypatch, change, match):
    args, manifest = fixture(tmp_path, monkeypatch)
    change(manifest)
    dump(args.input_root / "manifest.json", manifest)
    with pytest.raises(ValueError, match=match):
        mod.load_nominal(args.input_root, args.reference, args.model, args.legacy_mujoco_evaluator, args.cohort_protocol)


def test_payload_corruption_rejected_before_fk(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path, monkeypatch)
    path = args.input_root / manifest["cases"][0]["directory"] / "report.json"
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(mod.analysis, "load_fk", lambda *a: pytest.fail("Do not load FK after SHA failure"))
    with pytest.raises(ValueError, match="payload SHA/size"):
        mod.load_nominal(args.input_root, args.reference, args.model, args.legacy_mujoco_evaluator, args.cohort_protocol)


def test_training_contract_replaced_even_with_refreshed_local_ledger_is_rejected(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path, monkeypatch)
    path = args.input_root / manifest["training_metadata"]["physx"]["final_policy_contract"]
    value = json.loads(path.read_text()); value["physics_dt"] = .005; dump(path, value)
    refresh(args.input_root, manifest)
    with pytest.raises(ValueError, match="final contract differs"):
        mod.load_nominal(args.input_root, args.reference, args.model, args.legacy_mujoco_evaluator, args.cohort_protocol)


def test_relative_traversal_rejected(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path, monkeypatch)
    manifest["files"][0]["path"] = "../outside.json"
    dump(args.input_root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="stay relative"):
        mod.load_nominal(args.input_root, args.reference, args.model, args.legacy_mujoco_evaluator, args.cohort_protocol)


def test_indirect_payload_file_rejected(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path, monkeypatch)
    path = args.input_root / manifest["cases"][0]["directory"] / "report.json"
    target = tmp_path / "report.json"
    path.rename(target); path.symlink_to(target)
    with pytest.raises(ValueError, match="indirect payload"):
        mod.load_nominal(args.input_root, args.reference, args.model, args.legacy_mujoco_evaluator, args.cohort_protocol)


def test_changed_manifest_during_analysis_is_rejected(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path, monkeypatch)
    original = mod.analysis.analyze_case
    def changed(*values):
        dump(args.input_root / "manifest.json", {**manifest, "observed_utc": "changed"})
        return original(*values)
    monkeypatch.setattr(mod.analysis, "analyze_case", changed)
    with pytest.raises(ValueError, match="manifest changed during"):
        mod.load_nominal(args.input_root, args.reference, args.model, args.legacy_mujoco_evaluator, args.cohort_protocol)


def test_existing_output_preserved_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_nominal", lambda *a: pytest.fail("No payload loading before fresh output guard"))
    with pytest.raises(ValueError, match="Preserve"):
        mod.run(NS(output_dir=tmp_path))


def test_source_is_offline_and_no_complete_matrix_loader_or_policy_inference():
    source = Path(mod.__file__).read_text()
    for forbidden in ("torch.jit.load(", "mj_step(", "load_matrix(", "verify_cohort("):
        assert forbidden not in source
