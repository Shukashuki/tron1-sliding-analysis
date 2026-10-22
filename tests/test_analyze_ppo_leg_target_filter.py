"""Synthetic offline analysis; parent source-proof verification is injected."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_ppo_leg_target_filter as module
from test_analyze_ppo_ground_support_comparison import cohort, write_json, change_json


@pytest.fixture
def data(cohort, monkeypatch):
    root = cohort.tmp/"filter_cases"; root.mkdir()
    calls = []
    for label, source, alpha in (("baseline", cohort.baseline, 1.), ("candidate", cohort.candidate, .5)):
        folder = root/module.FOLDERS[label]; folder.mkdir()
        for name in ("physics_trajectory.npz", "report.json"):
            (folder/name).write_bytes((source/name).read_bytes())
        change_json(folder, "report.json", lambda doc: doc.update(checkpoint_sha256=module.parent.base.CHECKPOINT_SHA))
        change_json(folder, "report.json", lambda doc: doc.update(leg_target_filter_diagnostic={"alpha": alpha, "physical_gpu_index_requested": 1}))
        change_json(folder, "report.json", lambda doc: doc["policy_export"].update(checkpoint_sha256=module.parent.base.CHECKPOINT_SHA,
            external_leg_target_filter_alpha=alpha, actor_export_includes_leg_target_filter=False,
            leg_target_filter_diagnostic={"alpha": alpha, "physical_gpu_index_requested": 1}))
        time = np.arange(349)*.02
        requested = np.repeat((.2*np.sin(2*np.pi*20*time))[:, None], 6, axis=1)
        if alpha == 1.: filtered = requested.copy()
        else:
            filtered = np.empty_like(requested); previous = np.zeros(6)
            for index, target in enumerate(requested):
                previous = .5*previous+.5*target; filtered[index] = previous
        control = {"time_s": time+3., "sim_step_counter": np.arange(349)*4+12,
            "requested_targets_rad": requested, "filtered_targets_rad": filtered}
        physical = {"time_s": np.arange(1396)*.005+3., "control_index": np.repeat(np.arange(349), 4),
            "applied_targets_rad": np.repeat(filtered, 4, axis=0)}
        np.savez(folder/module.parent.evaluator.CONTROL_FILE, **control)
        np.savez(folder/module.parent.evaluator.PHYSICS_FILE, **physical)
        write_json(folder/module.parent.evaluator.RUNTIME_FILE, {"synthetic": True, "alpha": alpha})
    def verified(folder, alpha, *, physical_gpu_index):
        calls.append((folder.name, alpha, physical_gpu_index))
        return {"status": "verified", "alpha": alpha, "synthetic_injected_parent_boundary": True}
    monkeypatch.setattr(module.parent, "validate_output", verified)
    return SimpleNamespace(root=root, reference=cohort.reference, calls=calls, tmp=cohort.tmp)


def load(data): return module.load_comparison(data.root, data.reference)


def test_same_actor_parent_gates_identity_first_and_metrics_use_actual_target_sidecars(data):
    raw, summary = load(data)
    assert data.calls == [("alpha1p0", 1., 1), ("alpha0p5", .5, 1)]
    assert summary["physical_gpu_index_requested"] == 1 and summary["requested_device"] == "cuda:1"
    assert summary["same_frozen_actor"] and summary["case_order"] == ["baseline", "candidate"]
    assert summary["kind"] == "recorded_nominal_ppo_processed_leg_target_filter_comparison"
    assert summary["status"] == "requires_visual_and_tracking_review" and not summary["stable_contact_claimed"]
    assert summary["generic_requested_action_target_metrics_omitted"]
    for label in ("baseline", "candidate"):
        row = summary["cases"][label]
        assert row["checkpoint_sha256"] == module.parent.base.CHECKPOINT_SHA
        assert row["alpha"] == module.ALPHAS[label]
        assert row["target_clock_origins"]["absolute_native_control_time_s"] == 3.
        assert raw[label]["root_position_m"].shape == (1397, 1, 3)
        assert np.count_nonzero(raw[label]["actions_clipped"]) == 0
        core = row["windows"]["core"]
        assert "action" not in core  # Never mistake q0+rawaction for filtered target.
        assert core["target_control"]["filtered_applied_target_step_rms_rad"] > 0
        assert core["target_control"]["control_samples"] == 129
        assert core["target_control"]["target_change_pairs"] == 128
        assert core["target_control"]["physical_samples"] == 513
        assert core["tracking"]["xy_rmse_m"] == pytest.approx(.01)
        assert not row["windows"]["full"]["tracking"]["available"]
    changes = summary["relative_changes"]["core"]
    assert changes["target_control.requested_target_step_rms_rad"]["ratio"] == pytest.approx(1.)
    assert changes["target_control.filtered_applied_target_step_rms_rad"]["ratio"] < .5
    assert summary["cases"]["baseline"]["windows"]["core"]["target_control"]["requested_target_step_rms_rad"] == \
        summary["cases"]["baseline"]["windows"]["core"]["target_control"]["filtered_applied_target_step_rms_rad"]


def test_parent_rejection_prevents_downstream_analysis(data, monkeypatch):
    def rejected(folder, alpha, *, physical_gpu_index): raise ValueError("actual source/target/history proof failed")
    monkeypatch.setattr(module.parent, "validate_output", rejected)
    with pytest.raises(ValueError, match="proof failed"): load(data)


def test_wrong_parent_alpha_proof_rejected(data, monkeypatch):
    monkeypatch.setattr(module.parent, "validate_output", lambda *a, **kw: {"status": "verified", "alpha": .2})
    with pytest.raises(ValueError, match="alpha cohort"): load(data)


def test_changed_candidate_actor_is_not_a_filter_ablation(data):
    change_json(data.root/"alpha0p5", "report.json", lambda doc: doc.update(checkpoint_sha256="0"*64))
    change_json(data.root/"alpha0p5", "report.json", lambda doc: doc["policy_export"].update(checkpoint_sha256="0"*64))
    with pytest.raises(ValueError, match="SAME"): load(data)


@pytest.mark.parametrize("index", [1, 3, 4, 5])
def test_same_authorized_declared_gpu_is_passed_explicitly_to_full_parent_verifier(data, index):
    for label in module.CASE_ORDER:
        def declare(doc):
            doc["leg_target_filter_diagnostic"]["physical_gpu_index_requested"] = index
            doc["policy_export"]["leg_target_filter_diagnostic"]["physical_gpu_index_requested"] = index
        change_json(data.root/module.FOLDERS[label], "report.json", declare)
    _, summary = load(data)
    assert data.calls == [("alpha1p0", 1., index), ("alpha0p5", .5, index)]
    assert summary["physical_gpu_index_requested"] == index and summary["requested_device"] == f"cuda:{index}"


@pytest.mark.parametrize("value", [None, 0, 2, 6, True, "3"])
def test_missing_or_unauthorized_gpu_declaration_fails_before_parent_validation(data, value):
    change_json(data.root/"alpha0p5", "report.json", lambda doc: doc["leg_target_filter_diagnostic"].update(physical_gpu_index_requested=value))
    with pytest.raises(ValueError, match="explicitly declare an authorized"): load(data)
    assert data.calls == []


def test_different_authorized_cards_cannot_be_silently_pooled(data):
    change_json(data.root/"alpha0p5", "report.json", lambda doc: doc["leg_target_filter_diagnostic"].update(physical_gpu_index_requested=3))
    with pytest.raises(ValueError, match="SAME authorized"): load(data)
    assert data.calls == []


@pytest.mark.parametrize("change", ["kd", "actor_obs", "material"])
def test_shared_physical_or_observation_contract_must_match(data, change):
    def mutate(doc):
        if change == "kd": doc["policy_export"]["leg_kd"] = 5.
        if change == "actor_obs": doc["policy_export"]["extra_sensor"] = True
        if change == "material": doc["contact_material_audit"]["changed_material"] = True
    change_json(data.root/"alpha0p5", "report.json", mutate)
    with pytest.raises(ValueError): load(data)


def test_actual_target_hold_tampering_rejected_even_with_injected_parent_boundary(data):
    path = data.root/"alpha0p5"/module.parent.evaluator.PHYSICS_FILE
    with np.load(path) as archive: values = {name: archive[name] for name in archive.files}
    values["applied_targets_rad"][31, 0] += .1
    np.savez(path, **values)
    with pytest.raises(ValueError, match="hold each actual"): load(data)


def test_changed_target_control_clock_cannot_be_resampled(data):
    path = data.root/"alpha0p5"/module.parent.evaluator.CONTROL_FILE
    with np.load(path) as archive: values = {name: archive[name] for name in archive.files}
    values["time_s"][25] += .001
    np.savez(path, **values)
    with pytest.raises(ValueError, match="original349"): load(data)


def test_float32_step_accumulation_keeps_exact129_core_controls_without_changing_raw_arrays(data):
    raw_before, original = load(data)
    physical_sha = {label: module.smooth.sha256(data.root/module.FOLDERS[label]/"physics_trajectory.npz") for label in module.CASE_ORDER}
    step = float(np.float32(.005))
    expected_drift = abs(1392*(step-.005))
    for label in module.CASE_ORDER:
        folder = data.root/module.FOLDERS[label]
        path = folder/module.parent.evaluator.CONTROL_FILE
        control = module._archive(path)
        control["time_s"] = 3.+np.arange(349)*4*step
        np.savez(path, **control)
        path = folder/module.parent.evaluator.PHYSICS_FILE
        physics = module._archive(path)
        physics["time_s"] = 3.+np.arange(1396)*step
        np.savez(path, **physics)
    raw_after, result = load(data)
    for label in module.CASE_ORDER:
        before = original["cases"][label]["windows"]["core"]["target_control"]
        after = result["cases"][label]["windows"]["core"]["target_control"]
        assert after["control_samples"] == 129 and after["target_change_pairs"] == 128
        assert after["filtered_applied_target_step_rms_rad"] == before["filtered_applied_target_step_rms_rad"]
        assert after["native_control_float_clock_max_abs_deviation_from_schedule_s"] == pytest.approx(expected_drift, abs=1e-14)
        assert after["native_control_float_clock_verification_tolerance_s"] == 2e-6
        assert after["native_control_float_clock_preserved"]
        assert "Counter-derived nominal schedule" in after["target_clock"]
        assert np.array_equal(raw_after[label]["time_s"], raw_before[label]["time_s"])
        assert module.smooth.sha256(data.root/module.FOLDERS[label]/"physics_trajectory.npz") == physical_sha[label]
        saved = module._archive(data.root/module.FOLDERS[label]/module.parent.evaluator.CONTROL_FILE)
        assert np.array_equal(saved["time_s"], 3.+np.arange(349)*4*step)


@pytest.mark.parametrize("change", ["gap", "float_type"])
def test_nominal_schedule_still_requires_exact_integer_control_counters(data, change):
    path = data.root/"alpha0p5"/module.parent.evaluator.CONTROL_FILE
    values = module._archive(path)
    if change == "gap": values["sim_step_counter"][72] += 1
    else: values["sim_step_counter"] = values["sim_step_counter"].astype(float)
    np.savez(path, **values)
    with pytest.raises(ValueError, match="counters"): load(data)


def test_physical_torque_difference_uses_measured_leg_torques_not_requests(data):
    raw, _ = load(data)
    state = {key: value.copy() for key, value in raw["candidate"].items()}
    state["applied_joint_torque_nm"][:, 0, :6] = np.arange(1397)[:, None]*.5
    state["applied_joint_torque_nm"][:, 0, 6:] = 1.e6
    control = module._archive(data.root/"alpha0p5"/module.parent.evaluator.CONTROL_FILE)
    physics = module._archive(data.root/"alpha0p5"/module.parent.evaluator.PHYSICS_FILE)
    metrics = module.target_control_diagnostics(state, control, physics)
    assert metrics["core"]["physical_leg_applied_torque_step_rms_nm"] == pytest.approx(.5)
    assert metrics["full"]["physical_leg_applied_torque_step_rms_nm"] == pytest.approx(.5)


def test_source_states_not_modified_outputs_fresh_and_telemetry_kept_separately(data, monkeypatch):
    before = {label: module.smooth.sha256(data.root/module.FOLDERS[label]/"physics_trajectory.npz") for label in module.CASE_ORDER}
    output = data.tmp/"publication"
    monkeypatch.setattr(module, "render_contact_plot", lambda raw, summary, path: path.write_bytes(b"test plot only"))
    args = SimpleNamespace(input_root=data.root, reference=data.reference, output_dir=output)
    summary = module.run(args)
    assert set(summary["cases"]["candidate"]["input_telemetry_sha256"]) == {
        module.parent.evaluator.CONTROL_FILE, module.parent.evaluator.PHYSICS_FILE, module.parent.evaluator.RUNTIME_FILE}
    for label in module.CASE_ORDER:
        assert module.smooth.sha256(data.root/module.FOLDERS[label]/"physics_trajectory.npz") == before[label]
    with np.load(output/"measured_states.npz") as archive:
        assert not any("targets" in name or "actions" in name for name in archive.files)
        assert "candidate_joint_position_rad" in archive
    with pytest.raises(ValueError, match="fresh"): module.run(args)


def test_optional_input_root_default_and_required_fresh_output():
    args = module.parser().parse_args(["--output-dir", "new"])
    assert args.input_root == module.DEFAULT_INPUT
