"""Synthetic native200/400Hz inputs; no simulator, Torch or GPU required."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_ppo_contact_timestep as module


def write_json(path, value):
    path.write_text(json.dumps(value)+"\n")


@pytest.fixture
def data(tmp_path, monkeypatch):
    reference = tmp_path/"motion.npz"; rt = np.arange(349)*.02
    np.savez(reference, reference_time_s=rt, reference_position=np.column_stack((rt, np.zeros(349), np.full(349, .75))),
        reference_com_velocity=np.tile([1., 0., 0.], (349, 1)), reference_quaternion=np.tile([1., 0., 0., 0.], (349, 1)))
    monkeypatch.setattr(module, "MOTION_SHA256", module.smooth.sha256(reference))
    root = tmp_path/"cases"; root.mkdir(); calls = []
    joint_names = np.array([*module.LEG_NAMES, "wheel_L_Joint", "wheel_R_Joint"])
    for label in module.CASE_ORDER:
        folder = root/module.FOLDERS[label]; folder.mkdir()
        dt = module.PHYSICS_DT[label]; decimation = round(.02/dt); n = 349*decimation+1; time = np.arange(n)*dt
        counts = np.ones((n, 1, 2), dtype=np.int64); counts[::5, 0, 0] = 0
        force = np.zeros((n, 1, 2, 3)); force[..., 2] = 100.; force[::7, 0, 1, 2] = 3.
        omega = np.zeros((n, 1, 3)); omega[:, 0, 0] = (2. if label == "baseline" else 1.)*np.sin(2*np.pi*35*time)
        if label == "candidate": omega[:, 0, 0] += 3.*np.sin(2*np.pi*140*time)
        raw = {"time_s": time, "valid_mask": np.ones((n, 1), dtype=bool), "physics_dt_s": np.array(dt), "policy_dt_s": np.array(.02),
            "simulation_step_counter": np.arange(n)+24, "root_position_m": np.column_stack((time+.01, np.zeros(n), np.full(n, .75)))[:, None],
            "root_quaternion_wxyz": np.tile([1., 0., 0., 0.], (n, 1))[:, None],
            "root_linear_velocity_w_m_s": np.tile([1., 0., 0.], (n, 1))[:, None], "root_angular_velocity_w_rad_s": omega,
            "joint_position_rad": np.zeros((n, 1, 8)), "joint_velocity_rad_s": np.zeros((n, 1, 8)), "applied_joint_torque_nm": np.zeros((n, 1, 8)),
            "wheel_ground_contact_count": counts, "wheel_ground_contact_force_w_n": force,
            "actions_clipped": np.zeros((n, 1, 8)), "actions_raw": np.zeros((n, 1, 8)), "action_observation": np.zeros((n, 1, 51)),
            "action_was_applied": np.r_[False, np.ones(n-1, dtype=bool)][:, None],
            "reference_frame_for_action": np.r_[0, np.repeat(np.arange(349), decimation)][:, None],
            "joint_names": joint_names, "body_names": np.array(["base_Link", "wheel_L_Link", "wheel_R_Link"]),
            "wheel_body_names": np.array(["wheel_L_Link", "wheel_R_Link"])}
        np.savez(folder/"physics_trajectory.npz", **raw)
        report = {key: {"shared": key} for key in module.UNCHANGED_REPORT_FIELDS}
        report.update(status="evaluated", simulator="IsaacLab/PhysX", checkpoint_sha256=module.CHECKPOINT_SHA256,
            motion_file_sha256=module.MOTION_SHA256, physics_trajectory_sha256=module.smooth.sha256(folder/"physics_trajectory.npz"),
            evaluation_contact_friction_override={"evaluation_contact_friction": .7},
            contact_timestep_diagnostic={"physical_gpu_index_requested": 3, "physics_dt_s": dt},
            policy_export={"actor_dim": 51, "action_dim": 8, "physics_dt": dt, "decimation": decimation,
                "leg_kp": 500., "leg_kd": 10., "previous_action": "requested"})
        write_json(folder/"report.json", report)
    def verify(folder, physics_dt_s, physical_gpu_index=1):
        calls.append((folder.name, physics_dt_s, physical_gpu_index))
        return {"status": "verified", "physics_dt_s": physics_dt_s, "decimation": round(.02/physics_dt_s),
            "native_states": round(6.98/physics_dt_s)+1, "policy_decisions": 349, "native_clock_and_device_verified": True,
            "recording_sha256": module.smooth.sha256(folder/"physics_trajectory.npz"), "report_sha256": module.smooth.sha256(folder/"report.json")}
    parent = SimpleNamespace(validate_output=verify, __file__=__file__)
    monkeypatch.setitem(sys.modules, "run_ppo_contact_timestep", parent)
    return SimpleNamespace(root=root, reference=reference, parent=parent, calls=calls, tmp=tmp_path)


def change_report(data, label, update):
    path = data.root/module.FOLDERS[label]/"report.json"; report = module.read_json(path); update(report); write_json(path, report)


def change_raw(data, label, update):
    path = data.root/module.FOLDERS[label]/"physics_trajectory.npz"
    with np.load(path, allow_pickle=False) as z: raw = {k: z[k] for k in z.files}
    update(raw); np.savez(path, **raw)
    change_report(data, label, lambda r: r.update(physics_trajectory_sha256=module.smooth.sha256(path)))


def load(data): return module.load_comparison(data.root, data.reference)


def test_native_clocks_counts_and50hz_context_preserved_without_resampling(data):
    raw, summary = load(data)
    assert data.calls == [("dt0p005", .005, 3), ("dt0p0025", .0025, 3)]
    for label, n, core, stride in (("baseline", 1397, 513, 4), ("candidate", 2793, 1025, 8)):
        case = summary["cases"][label]
        assert len(raw[label]["time_s"]) == case["physical_samples"] == n
        assert case["decimation"] == stride and case["control_decisions"] == 349
        assert case["windows"]["core"]["physical_samples"] == core
        assert case["windows"]["core"]["tracking"]["xy_rmse_m"] == pytest.approx(.01)
        assert case["windows"]["core"]["tracking"]["speed_retention"] == 1.
        assert case["windows"]["full"]["tracking"]["available"] is False
        assert case["reference_overlap_tracking"]["available"] is True
        assert case["reference_overlap_tracking"]["interval_s"] == [0., 6.96]
    assert summary["same_frozen_actor"] and not summary["stable_contact_claimed"] and not summary["height_or_flight_failure_gate"]


def test_shared_band_improves_but_extra400hz_band_makes_native_total_larger(data):
    _, summary = load(data)
    before, after = [summary["cases"][label]["windows"]["core"]["spectra_common_band10_100"]["world_angular_xy"] for label in module.CASE_ORDER]
    assert before["native_nyquist_hz"] == 100. and after["native_nyquist_hz"] == 200.
    assert after["common_band_rms"]/before["common_band_rms"] == pytest.approx(.5, rel=.005)
    assert after["native_high_frequency_rms"] > before["native_high_frequency_rms"]
    assert summary["relative_common_band_changes"]["core"]["world_angular_xy_common_band_rms"]["ratio"] == pytest.approx(.5, rel=.005)


@pytest.mark.parametrize("dt", [.005, .0025])
def test_harmonic_common_band_parseval_channel_mean_and_no_dc(dt):
    time = np.arange(round(4/dt))*dt
    values = np.column_stack((2*np.sin(2*np.pi*25*time)+3.+time, np.zeros(len(time))))
    spectrum = module.band_spectrum(values, dt)
    assert spectrum["common_band_mean_square"] == pytest.approx(1., rel=1e-4)
    assert spectrum["native_high_frequency_rms"] == pytest.approx(1., rel=1e-4)
    assert module.band_spectrum(np.zeros((len(time), 2)), dt)["common_band_rms"] == 0.


def test_contact_count_and_load_eligibility_are_distinct_not_flight(data):
    _, summary = load(data)
    core = summary["cases"]["candidate"]["windows"]["core"]
    measured = core["contact"]["measured_force_bearing_contact"]
    load_gate = core["contact"]["load_eligibility"]
    assert measured["right"]["sample_duty_fraction"] == 1.
    assert load_gate["right"]["sample_duty_fraction"] < 1.
    assert load_gate["right"]["state_switches_per_s"] > 0
    assert core["native_dt_s"] == .0025 and core["tilt"]["not_a_failure_gate"]
    assert "different native dt" in core["step_difference_comparison_note"]


@pytest.mark.parametrize("value", [None, True, 2, "3", 6])
def test_missing_or_unauthorized_gpu_before_parent_validation(data, value):
    change_report(data, "candidate", lambda r: r["contact_timestep_diagnostic"].update(physical_gpu_index_requested=value))
    with pytest.raises(ValueError, match="SAME authorized"): load(data)
    assert not data.calls


def test_different_declared_gpu_cannot_be_pooled(data):
    change_report(data, "candidate", lambda r: r["contact_timestep_diagnostic"].update(physical_gpu_index_requested=4))
    with pytest.raises(ValueError, match="SAME authorized"): load(data)


def test_parent_reject_and_wrong_native_cohort_proof_fail_closed(data):
    data.parent.validate_output = lambda *a, **kw: {"status": "verified", "physics_dt_s": .005}
    with pytest.raises(ValueError, match="Parent clock/device/source"): load(data)


@pytest.mark.parametrize("change", ["checkpoint", "mu", "kd", "material"])
def test_actor_material_gain_and_source_contract_not_silently_changed(data, change):
    def mutate(r):
        if change == "checkpoint": r["checkpoint_sha256"] = "0"*64
        elif change == "mu": r["evaluation_contact_friction_override"]["evaluation_contact_friction"] = .8
        elif change == "kd": r["policy_export"]["leg_kd"] = 5.
        else: r["contact_material_audit"] = {"different": True}
    change_report(data, "candidate", mutate)
    with pytest.raises(ValueError): load(data)


@pytest.mark.parametrize("change", ["time", "counter", "valid", "observation", "count", "action"])
def test_native_time_type_context_and_contact_guards(data, change):
    def mutate(r):
        if change == "time": r["time_s"][10] += .0001
        elif change == "counter": r["simulation_step_counter"][10] += 1
        elif change == "valid": r["valid_mask"][10] = False
        elif change == "observation": r["action_observation"][9, 0, -1] = .1
        elif change == "count": r["wheel_ground_contact_count"] = r["wheel_ground_contact_count"].astype(float)
        else: r["actions_clipped"][10, 0, 0] = .1
    change_raw(data, "candidate", mutate)
    with pytest.raises(ValueError): load(data)


def test_reference_sha_and_report_native_binding_are_required(data):
    with pytest.raises(ValueError, match="Locked reference"):
        module.load_comparison(data.root, data.root/"dt0p005"/"physics_trajectory.npz")
    change_report(data, "candidate", lambda r: r.update(physics_trajectory_sha256="0"*64))
    with pytest.raises(ValueError, match="hash binding"): load(data)


def test_loading_and_export_copy_preserve_all_native_inputs(data):
    paths = list(data.root.rglob("*.npz"))+list(data.root.rglob("*.json"))+[data.reference]
    hashes = {str(p): module.smooth.sha256(p) for p in paths}
    raw, _ = load(data)
    arrays = module.measured_state_arrays(raw)
    assert arrays["baseline__time_s"].shape == (1397,) and arrays["candidate__time_s"].shape == (2793,)
    arrays["candidate__time_s"][10] = -10.
    assert raw["candidate"]["time_s"][10] == .025
    assert {str(p): module.smooth.sha256(p) for p in paths} == hashes


def test_fresh_output_guard_before_any_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "load_comparison", lambda *a: pytest.fail("Should not load"))
    args = module.parser().parse_args(["--input-root", "missing", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="fresh"): module.run(args)


def test_publication_writes_original_native_arrays_and_hashes(data, monkeypatch):
    output = data.tmp/"publication"
    monkeypatch.setattr(module, "render_contact_plot", lambda raw, summary, path: path.write_bytes(b"synthetic CPU plot placeholder"))
    args = module.parser().parse_args(["--input-root", str(data.root), "--reference", str(data.reference), "--output-dir", str(output)])
    summary = module.run(args)
    assert json.loads((output/"summary.json").read_text())["same_frozen_actor"]
    with np.load(output/"measured_states.npz", allow_pickle=False) as z:
        assert z["baseline__time_s"].shape == (1397,) and z["candidate__time_s"].shape == (2793,)
    for artifact in summary["artifacts"].values():
        assert module.smooth.sha256(output/artifact["file"]) == artifact["sha256"]
    with pytest.raises(ValueError, match="fresh"): module.run(args)
