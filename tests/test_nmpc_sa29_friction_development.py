"""Separate SA29 friction-development audits; synthetic files, no simulation."""
import copy
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_nmpc_sa29_friction_development as analysis
import analyze_nmpc_sa29_pilot as pilot
from analyze_controller_pair import posture_diagnostic
from analyze_nmpc_nominal_development import contact_diagnostic
from analyze_tvlqr_pilot import PUBLIC_SIGNALS, PUBLIC_TRACE_FIELDS
from controller_references import get_reference
from test_nmpc_nominal_development import canonical, geometry
from test_nmpc_sa29_pilot import RAW_IDENTITY_SHA256, mutate, public_trace, sa29_raw, sa29_report


REFERENCE_ID = "doa_sa29_scale2"
FRAMES, SAMPLES = 258, 1033
GRID = [.01, .03, .05, .1, .15, .2, .3, .45, .6, .7, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.]
PREDECLARATION_SHA256 = "c6b3397431ebcb22116c6dde60e78f0d15bd0a5750511b7dcd438a49fa59a62c"
PUBLIC_SHA256 = {
    "summary.json": "386004dce4309a9d95114759f71b19486be38cc0b6498d6187b686846438669a",
    "development_trace.npz": "4d0cbe38eeea069eb773404abb1d966a91db49c627d350ed5a72fdefbc525388",
    "topdown.png": "1d3a7971a0eb7c03a340a7297390ec37cc670045b0c5262c443140baac5d37df"}
ANALYZER_SHA256 = "39de9e30ef51e013750eb30955fc771a63833d2a4f01b81e89f842e1a3e726c5"
PUBLIC_RAW_IDENTITY_SHA256 = {
    "mu0p010": {
        "report": "4ff3da9e1cf49d8f08f5f6eb59c234d2b42f7e82e5c15ffa8edb82c57c4be201",
        "trace": "65a59d9149ae89304864987ea4442ff68aa9a42041f3d9e620a2dac46002885e",
        "solver": "2cca193332f59197c497448f53aeee58058aab9b8f03e5fe4113661f84b87db3"
    },
    "mu0p030": {
        "report": "adc7e53ef00207104e283b690fa1b7dfc908862d914be0c51153d9637097574d",
        "trace": "0939c44b60794e9d242dbf1771c2c9e6c94527552117c064850c1926dba12910",
        "solver": "644b17209bc1b8d9eeeb5ebc6393b789b9bf664279eb0ecc3249c3c9918f51f7"
    },
    "mu0p050": {
        "report": "44f09984dcb5f940479526df3262e16204a2feedd52f317f239bdbd446f20d08",
        "trace": "eb7124d429bcff721b842916cd75464cf4c234f5e8dfa398c1a6a8953ddc124f",
        "solver": "9715f4a88e9a400828f9a274aa655e6e52396c6372b17c785f504fe01b43861c"
    },
    "mu0p100": {
        "report": "dcac20071965ef08122541d63221aa273193e14ccef9c5dac7a655cb43d0780b",
        "trace": "7670a38d558ed7eeedc8d78db7d430816ee247b4a1e0ff56c785e139350d4e61",
        "solver": "73836ae69b3096c7bf1c6f50235db1c9927adf1738086ad7feec3d5cd472a566"
    },
    "mu0p150": {
        "report": "4ab92410f2dbc3611ef4251204d2b25659939992911bc1b6975867152c948e15",
        "trace": "c9ebfb28c00ff3f8cb2a86b584ba0e94b984def514b1e7b94a5b29b7f0637a3a",
        "solver": "a27d8134c3a4e20f68be1342e6e01a7ebf45ff876a273afe1060dfd59e01eb9d"
    },
    "mu0p200": {
        "report": "f33811e4460dd2b0eb22feb1b0db528b068b29a93ca7450ea133ef9b85283b98",
        "trace": "c5050a852a92144f10219cb95dc5a28a8e465675aff5a571f824b71683a85df1",
        "solver": "8144fa67031c5b8e08cd094e3cfe879959d7fead92e238d2d1183e24ba8cd812"
    },
    "mu0p300": {
        "report": "cf249f896479f672598606bcfd8e201f4ac1484d560c842940f6a9704127e660",
        "trace": "ab659d9b96da48ef3bb354e6dc94db261317438fc0572d6ee943a1f69e70055d",
        "solver": "ad6ff004721d4c2b2800aebd6f6fb6135cbba3f1b60c905642a53531702d4622"
    },
    "mu0p450": {
        "report": "b7597781a97f008a404a7cae5f1ff289359afbc2c4548451343abb3aede36078",
        "trace": "017cf75fea1098ec3657bbb06ae6584bf41ed55b833fd091bcd830478ce0a925",
        "solver": "177065c111f74048c84bb5ad6b86ce8d77fef016e86f7941a3a24cd86dc7f7f8"
    },
    "mu0p600": {
        "report": "e2d9582bc6342d166b437afb739428c7454e153ccdd1e3c3acfdff13c2bd9254",
        "trace": "3acaa7821afc623fa4c7e1ab3efb35ca44fd8582c33040444c283e79dc1b9cd1",
        "solver": "eb72915eea5827503070fc64995bf27b94ca16491906ea64bd450da3a122fda9"
    },
    "mu0p700": {
        "report": "fe703e078fce33c79670d0bae14a3966b21c03b7a68d5ae90871f55374ec9356",
        "trace": "aecb80f386f43c831664e8e3f41b678c15551922f9c4147d92475573e827a388",
        "solver": "0b101dd0dac648566068575836e9cbea6333fdf6e942c6764dd94b9b4a37c725"
    },
    "mu0p800": {
        "report": "03b23798fd8f29cbced615a7b111e4074c75f00c9fecc69835d6ceaf54b7fe4a",
        "trace": "83ae110a9c68898afe125870aa4f2f4d9dfa9075e028f2b137e9054fd27c03f6",
        "solver": "8a1f613eadb10079b9df61f68225c15c9ab2e4dcb6e0e04a903435cd26f09760"
    },
    "mu1p000": {
        "report": "6a6617690fb873542def0385f2094ca8db265094834271f613ba4b754e2882f4",
        "trace": "33f9d417b7eac5fc92b4cc31d4105013f008dee2463749b66128dbcf89217fbd",
        "solver": "4d547f452c47877fb00af66fe6972b47672cdc3b950f076c275744d5b688a103"
    },
    "mu1p200": {
        "report": "a228800742e7f1e0049a72087435b49d6468bea22612a9c67ca259ae13502c94",
        "trace": "237e558a44de684d78570bd3b0767e2026ce9499b7e6da28a6e263936f520dab",
        "solver": "7c444217685569b2d0308b05f7a146f8d4f96ee11ef202c85209af74b6078a40"
    },
    "mu1p400": {
        "report": "11e94f32f951841829616654bc34efbe341144c85b0f0b23678da1bf067160ca",
        "trace": "5fe97264d225d44df8afef93ce95f0f3522d2ee8851fbd124e7164a2bf829618",
        "solver": "8caffff6ffcf2b28feac660da51ce6f311daec52f928d681758659de784760f0"
    },
    "mu1p600": {
        "report": "5ea850f86dda2b8b3df348ed70cf593fef8c1939c926b5aaa09be2614a0930f1",
        "trace": "ed6135d4d296dd82754a27fed521ad221c5fb8bff7616970d13589abe3509b8b",
        "solver": "3d02ce1cc71c21a8bc31668ffec7e3fdd1b4360e89ee2b3bc166fb8ec8f5fd91"
    },
    "mu1p800": {
        "report": "d8900bc4717d98a7c2d65f4992a346c63a10da9a3ee993ca7c052c9fd0c28d25",
        "trace": "c6f8ad22263e4be88cdfdd1208fef6002ba80cabfde5dec1a695f3ffff94edc9",
        "solver": "082fdd621303ba06f0df4308c2bbca47ce74d9c1085ec1d5a7f22a41edec45a8"
    },
    "mu2p000": {
        "report": "ccd936e6f5ff14786ecc6f64811faae69a0ceaff2dbbf193898ab29f10a4cd3a",
        "trace": "5b41d5f69858ba6f6bd48957d9f5f8410026a953e68f89b45d387c708a4bf30c",
        "solver": "eda8ce22af8fe0899085a3f04b1eacc676844ab7b9f2c57024f3e5caedea37e3"
    }
}


def declaration_fixture():
    return json.loads((ROOT / "config/nmpc_sa29_friction_development.json").read_text())


def report_for(mu):
    report = sa29_report()
    report["plant_mu"] = mu
    report["plant_material_readback"]["compiled_sliding_friction_range"] = [mu, mu]
    return report


def header(report, mu):
    return analysis.audit_header(report, mu, get_reference(REFERENCE_ID),
        model_sha256=declaration_fixture()["input_sha256"]["model"])


def write_case(directory, mu, *, status="completed", completed=FRAMES):
    """Hash-bind original synthetic files without loading a robot or stepping."""
    directory.mkdir()
    report = report_for(mu)
    raw, log = sa29_raw()
    np.savez_compressed(directory / "rollout.npz", **raw)
    (directory / "solver_log.json").write_text(json.dumps(log))
    report.update(status=status, completed_policy_steps=completed,
        rollout_sha256=analysis.sha256(directory / "rollout.npz"),
        solver_log_sha256=analysis.sha256(directory / "solver_log.json"))
    (directory / "report.json").write_text(json.dumps(report))
    return report


def save_report(directory, report):
    (directory / "report.json").write_text(json.dumps(report))


def test_predeclared_17_case_supplement_keeps_primary_runtime_but_never_primary_reference():
    manifest = declaration_fixture()
    primary = json.loads((ROOT / "config/nmpc_friction_development.json").read_text())
    assert analysis.audit_predeclaration() == manifest
    assert analysis.PREDECLARATION_SHA256 == PREDECLARATION_SHA256
    assert hashlib.sha256(analysis.PREDECLARATION.read_bytes()).hexdigest() == PREDECLARATION_SHA256
    assert analysis.expected_grid() == GRID == manifest["plant_mu"]
    assert manifest["source_sha256"] == primary["source_sha256"] and len(manifest["source_sha256"]) == 21
    assert manifest["reference_id"] == REFERENCE_ID != primary["reference_id"]
    assert manifest["role"] == "separate_supplement" and manifest["stratum"] == "high_sideslip_s_avoidance"
    assert manifest["intended_local_episodes"] == 17 and manifest["new_local_episodes"] == 16
    assert manifest["reused_existing_plant_mu"] == [.7]
    assert manifest["evaluation"] == {
        "frames": FRAMES, "physical_samples": SAMPLES, "control_dt_s": .02, "physics_dt_s": .005,
        "duration_s": 5.16, "core_window_s": [0, 3.14],
        "supplementary_event_windows_s": [[.78, 1.54], [1.88, 3.04]],
        "events_are_descriptive_not_gates": True, "xy_screen_limits_m": {"rmse": .15, "peak": .3}}
    assert analysis.REUSED_NOMINAL_INPUT_SHA256 == RAW_IDENTITY_SHA256
    for key in ("pooled_success_statistics", "retuning_or_selection_after_friction_results",
        "formal_controller_freeze_claimed", "qualified_validation_driver_modified",
        "isaac_results_for_this_development_sweep", "sim2sim_sweep_claimed",
        "height_flight_or_heading_offset_failure_gate"):
        assert manifest[key] is False


@pytest.mark.parametrize("path,value", [
    (("reference_id",), "drift_drl_traj4"), (("pooled_success_statistics",), True),
    (("plant_mu",), [.1, .7]), (("reused_existing_plant_mu",), []),
    (("prediction_mu_scenarios",), [.1]), (("explicit_overrides", "iterations"), 4),
    (("evaluation", "frames"), 349), (("evaluation", "duration_s"), 6.98),
    (("retuning_or_selection_after_friction_results",), True),
    (("isaac_results_for_this_development_sweep",), True),
    (("source_sha256", "scripts/nmpc_shooting.py"), "0" * 64),
])
def test_predeclaration_cannot_be_changed_after_observing_friction_results(path, value):
    manifest = declaration_fixture(); mutate(manifest, path, value)
    with pytest.raises(ValueError):
        analysis.audit_predeclaration(manifest)


@pytest.mark.parametrize("mu", GRID)
def test_header_audits_real_plant_coefficient_without_normalizing_to_nominal(mu):
    report = report_for(mu); before = copy.deepcopy(report)
    public = header(report, mu)
    assert report == before
    assert public["actual_plant_mu"] == mu
    assert public["plant_material_readback"]["compiled_sliding_friction_range"] == [mu, mu]
    assert public["prediction_mu_scenarios"] == [.7]
    assert public["runtime_source_sha256"] == declaration_fixture()["source_sha256"]
    payload = json.dumps(public)
    assert "/private/" not in payload and "compiler_command" not in payload and "worker_pid" not in payload


def test_generic_header_receives_the_original_non_nominal_report(monkeypatch):
    report = report_for(.03)
    original = analysis.audit_fixed_header
    seen = []
    def observed(actual, mu, spec, model_sha256):
        seen.append((actual is report, actual["plant_mu"], mu,
            actual["plant_material_readback"]["compiled_sliding_friction_range"]))
        return original(actual, mu, spec, model_sha256)
    monkeypatch.setattr(analysis, "audit_fixed_header", observed)
    header(report, .03)
    assert seen == [(True, .03, .03, [.03, .03])]


@pytest.mark.parametrize("path,value", [
    (("plant_mu",), .7), (("plant_material_readback", "compiled_sliding_friction_range"), [.7, .7]),
    (("reference_id",), "drift_drl_traj4"), (("role",), "primary"),
    (("pooled_success_statistics",), True), (("prediction_mu_scenarios",), [.1]),
    (("effective_cem_config", "iterations"), 4), (("model_sha256",), "0" * 64),
    (("settings", "nmpc", "horizon_steps"), 30), (("learned_policy_used",), True),
    (("isaac_result",), True), (("pose_height_failure_gate",), True),
    (("real_time_verified",), True), (("robustness_certificate",), True),
    (("action_parameterization", "learned_baseline"), True),
    (("source_sha256", "/private/archived/scripts/nmpc_shooting.py"), "0" * 64),
    (("native_prediction_backend", "workers", "0.7", "native_library_sha256"), "0" * 64),
])
def test_changed_coefficient_identity_controller_or_scope_fails_closed(path, value):
    report = report_for(.1); mutate(report, path, value)
    with pytest.raises(ValueError):
        header(report, .1)


def test_complete_non_nominal_episode_replays_258_actions_1033_samples_and_common_score(tmp_path):
    report = write_case(tmp_path / "mu0p100", .1)
    evidence, trace, public = analysis.audit_episode(tmp_path / "mu0p100", .1,
        get_reference(REFERENCE_ID), report["model_sha256"])
    assert evidence["plant_mu"] == public["actual_plant_mu"] == .1
    assert public["complete_physical_evidence"] is True and public["censored"] is False
    assert trace["time_s"].shape == (SAMPLES,) and trace["time_s"][-1] == 5.16
    assert public["solver_diagnostics"]["decision_samples"] == FRAMES
    counts = public["solver_diagnostics"]["counts"]
    assert counts["candidate_evaluations"] == counts["finite_candidates"] == counts["scenario_evaluations"] == 50309
    assert counts["batch_prediction_calls"] == 1547 and counts["warm_start_used"] == 257
    assert public["solver_diagnostics"]["common_score_arithmetic_verified"] is True
    assert public["simulation_budget"] == {"actual_plant_physics_substeps": 1032,
        "requested_prediction_substeps": 6037080, "failed_prediction_requests_prove_completed_steps": False}
    assert public["solver_diagnostics"]["timing"]["real_time_verified"] is False
    assert public["solver_diagnostics"]["timing"]["actuation_delay_injected"] is False


def test_nominal_reuse_rejects_a_new_rerun_even_if_its_controller_and_metrics_match(tmp_path):
    report = write_case(tmp_path / "nominal-rerun", .7)
    with pytest.raises(ValueError, match="(?i)(nominal|reuse|original|hash)"):
        analysis.audit_episode(tmp_path / "nominal-rerun", .7, get_reference(REFERENCE_ID), report["model_sha256"])


@pytest.mark.parametrize("status", ["failed", "running"])
def test_incomplete_reports_are_censored_even_when_full_synthetic_archives_exist(tmp_path, status):
    report = write_case(tmp_path / "incomplete", .3, status=status, completed=12)
    evidence, trace, public = analysis.audit_episode(tmp_path / "incomplete", .3,
        get_reference(REFERENCE_ID), report["model_sha256"])
    assert evidence is None and trace is None
    assert public["censored"] is True and public["complete_physical_evidence"] is False
    assert public["physical_xy_failure_claimed"] is False
    assert public["reported_completed_policy_steps"] == 12
    assert set(public["input_sha256"]) == {"report"}
    assert "error" not in public and "solver_diagnostics" not in public


@pytest.mark.parametrize("problem", ["hash", "phase", "action", "state", "clock", "core", "stage", "decision_count"])
def test_completed_episode_requires_original_clock_phase_state_hash_and_score(tmp_path, problem):
    directory = tmp_path / "case"; report = write_case(directory, .1)
    raw = analysis.load_arrays(directory / "rollout.npz")
    if problem == "hash": report["rollout_sha256"] = "0" * 64
    elif problem == "stage": report["total_stage_cost"] += 1.
    elif problem == "decision_count": report["completed_policy_steps"] = 257
    else:
        if problem == "phase": raw["reference_frame_for_action"][100] += 1
        elif problem == "action": raw["held_action"][100, 0] += .1
        elif problem == "state": raw["qpos"][100, 0] += .1
        elif problem == "clock": raw["time_s"][100:] += .005
        else: raw["reference_evaluation_core_interval_s"][1] = 2.
        np.savez_compressed(directory / "rollout.npz", **raw)
        report["rollout_sha256"] = analysis.sha256(directory / "rollout.npz")
    save_report(directory, report)
    with pytest.raises(ValueError):
        analysis.audit_episode(directory, .1, get_reference(REFERENCE_ID), report["model_sha256"])


def test_metrics_keep_speed_heading_contacts_posture_and_event_windows_separate_from_xy_failure():
    raw, log = sa29_raw()
    native = pilot.audit_trace(raw, log, get_reference(REFERENCE_ID))
    native["quaternion"][:] = [np.cos(np.pi / 6), 0., 0., np.sin(np.pi / 6)]
    native["contact_slip"][:, 0] = 3.
    traces = {"reference": canonical(FRAMES, .02), analysis.case_name(.1): native}
    recon = {name: geometry(trace) for name, trace in traces.items()}
    metrics = analysis.summarize_cases(traces, recon, get_reference(REFERENCE_ID))
    case = analysis.case_name(.1)
    windows = [(metrics["core"], 629)] + [(event["metrics"], count)
        for event, count in zip(metrics["supplementary_events"], (153, 233))]
    for values, count in windows:
        item = values[case]
        assert item["samples_in_window"] == count and item["coverage_complete"] is True
        assert item["reference_xy_rmse_m"] == pytest.approx(0., abs=1e-14)
        assert item["large_xy_exploratory_screen"] is False
        assert item["mean_speed_retention_ratio"] == pytest.approx(1.)
        assert item["heading_geometry"]["root_directed_beta"]["mean_absolute_deg"] == pytest.approx(60.)
        assert item["ground_contact_lateral_slip_m_s"] == {"left": 3., "right": None}
        assert item["heading_offset_is_mismatch"] is item["height_or_flight_failure_gate"] is False
        assert values["reference"]["ground_contact_heading_geometry"] is None
        assert values["reference"]["ground_contact_lateral_slip_m_s"] is None
    assert [item["interval_s"] for item in metrics["supplementary_events"]] == [[.78, 1.54], [1.88, 3.04]]
    assert all(item["descriptive_not_gate"] is True for item in metrics["supplementary_events"])
    assert "not a height/flight failure gate" in posture_diagnostic(native)["role"]


def test_strict_force_and_speed_masks_do_not_turn_airborne_zeros_into_rolling():
    native = canonical(SAMPLES, .005)
    native["contact_force"][:, 0, 2] = 5.
    native["contact_slip"][:] = 0.
    native["com_velocity"][100:150] = 0.
    traces = {"reference": canonical(FRAMES, .02), analysis.case_name(.1): native}
    recon = {name: geometry(trace) for name, trace in traces.items()}
    value = analysis.summarize_cases(traces, recon, get_reference(REFERENCE_ID))["core"][analysis.case_name(.1)]
    assert value["ground_contact_heading_geometry"]["contact_fraction_left_right"] == [0., 0.]
    assert value["ground_contact_lateral_slip_m_s"] == {"left": None, "right": None}
    assert value["heading_geometry"]["root_directed_beta"]["valid_samples"] == 579
    assert value["ground_contact_heading_geometry"]["per_wheel"]["left"]["root_course_to_wheel_gamma"]["valid_samples"] == 0


def test_parse_runs_accepts_one_case_but_rejects_duplicates_missing_paths_and_off_grid_values(tmp_path):
    first = tmp_path / "a"; first.mkdir()
    second = tmp_path / "b"; second.mkdir()
    assert analysis.parse_runs([f".1={first}"]) == {.1: first}
    assert list(analysis.parse_runs([f".7={second}", f".1={first}"])) == [.1, .7]
    for values in ([], [f".1={first}", f".1={second}"], [f".1={first}", f".3={first}"],
        [f".11={first}"], [f"nan={first}"], [f".3={tmp_path / 'absent'}"], [".1"]):
        with pytest.raises(ValueError):
            analysis.parse_runs(values)


def test_cli_has_no_primary_policy_or_isaac_inputs_and_preserves_existing_output(tmp_path):
    args = analysis.parser().parse_args(["--motion", "m", "--reference-file", "r", "--model", "x",
        "--source-root", "s", "--run", ".1=d", "--output-dir", "o"])
    assert args.motion_file == Path("m") and args.run == [".1=d"]
    assert not any(hasattr(args, name) for name in ("ppo_grid", "sac_grid", "isaac_grid"))
    with pytest.raises(ValueError, match="Preserve existing"):
        analysis.run(SimpleNamespace(output_dir=tmp_path))


def test_partial_window_and_missing_cases_remain_censored_not_xy_success_or_failure():
    traces = {"reference": canonical(FRAMES, .02), analysis.case_name(.1): canonical(600, .005)}
    recon = {name: geometry(trace) for name, trace in traces.items()}
    metrics = analysis.summarize_cases(traces, recon, get_reference(REFERENCE_ID))
    value = metrics["core"][analysis.case_name(.1)]
    assert value["coverage_complete"] is False
    assert value["reference_xy_rmse_m"] is value["reference_xy_peak_m"] is value["large_xy_exploratory_screen"] is None
    rows = analysis.make_rows(metrics, {analysis.case_name(.1): {"censored": False}})
    assert len(rows) == 17 and [row["mu"] for row in rows] == GRID
    missing = next(row for row in rows if row["mu"] == .3)
    assert missing["censored"] is True
    assert missing["NMPC"]["reference_xy_rmse_m"] is missing["NMPC"]["mean_speed_retention_ratio"] is None
    assert all(item["descriptive_not_gate"] is True and item["NMPC"]["censored"] is True
        for item in missing["supplementary_events"])
    with pytest.raises(ValueError, match="censored"):
        analysis.make_rows(metrics, {analysis.case_name(.1): {"censored": True}})


def fixture_arguments(tmp_path, monkeypatch, *, complete):
    """Exercise actual file decoders/auditors, replacing only locked assets/FK.

    A complete synthetic grid explicitly replaces the nominal *fixture* archive
    identity. Production source still pins the genuine prior pilot, tested above.
    No dynamics, original model loading, native library or policy is used here.
    """
    reference = canonical(FRAMES, .02)
    motion = {"time_s": reference["time_s"], "fps": np.array([50.]),
        "joint_names": np.array(pilot.JOINT_NAMES),
        "body_names": np.array(["base_Link", "wheel_L_Link", "wheel_R_Link"]),
        "body_pos_w": np.concatenate((reference["position"][:, None], reference["wheel_pos"]), axis=1),
        "body_quat_w": np.repeat(reference["quaternion"][:, None], 3, axis=1),
        "body_lin_vel_w": np.repeat(reference["com_velocity"][:, None], 3, axis=1),
        "body_ang_vel_w": np.repeat(reference["angular_velocity"][:, None], 3, axis=1),
        "joint_pos": reference["joint_pos"], "joint_vel": reference["joint_vel"]}
    motion_file, reference_file, model = [tmp_path / name for name in ("motion.npz", "reference.npz", "model.xml")]
    np.savez(motion_file, **motion)
    np.savez(reference_file, time_s=reference["time_s"], root_target_position_m=reference["position"])
    model.write_bytes(b"Synthetic asset token; never opened by a physics engine")
    declaration = declaration_fixture()
    locked = {path: declaration["input_sha256"][name]
        for path, name in ((motion_file, "motion"), (reference_file, "robot_reference"), (model, "model"))}
    original_sha256 = analysis.sha256
    monkeypatch.setattr(analysis, "sha256", lambda path: locked.get(Path(path)) or original_sha256(path))
    monkeypatch.setattr(analysis, "verify_reference_pair", lambda *unused: get_reference(REFERENCE_ID))
    args = analysis.parser().parse_args(["--motion", str(motion_file), "--reference-file", str(reference_file),
        "--model", str(model), "--source-root", str(tmp_path / "source"), "--run", "placeholder",
        "--output-dir", str(tmp_path / "new-output")])
    args.run = []
    for mu in (GRID if complete else [.1, .3]):
        directory = tmp_path / analysis.case_name(mu)
        report = write_case(directory, mu, status="failed" if not complete and mu == .3 else "completed", completed=12 if not complete and mu == .3 else FRAMES)
        args.run.append(f"{mu}={directory}")
        if mu == .7:
            monkeypatch.setattr(analysis, "REUSED_NOMINAL_INPUT_SHA256", {
                "report": original_sha256(directory / "report.json"),
                "trace": original_sha256(directory / "rollout.npz"),
                "solver": original_sha256(directory / "solver_log.json")})
    helpers = {name: analysis.source_digest(report, name) for name in pilot.FK_HELPERS}
    @contextmanager
    def loader(*unused):
        yield None, {"dynamics_stepping": False, "source_helpers_sha256": helpers}
    class FK:
        def __init__(self, unused): pass
        def reconstruct(self, trace): return geometry(trace)
    monkeypatch.setattr(analysis, "source_model", loader)
    monkeypatch.setattr(analysis, "WheelKinematics", FK)
    monkeypatch.setattr(analysis, "render_topdown", lambda compact, summary, output: Path(output).write_bytes(b"synthetic-render-token"))
    return args


def assert_portable(summary, arrays):
    allowed = set(PUBLIC_TRACE_FIELDS) | set(PUBLIC_SIGNALS) | {"wheel_quaternion_wxyz", "wheel_center_velocity_w"}
    labels = {"reference", *[analysis.case_name(mu) for mu in GRID]}
    for key, value in arrays.items():
        if key not in {"schema_version", "wheel_order"}:
            prefix, separator, field = key.partition("_")
            assert separator and prefix in labels and field in allowed
        assert value.dtype.kind != "O"
    assert "reference_contact_count" not in arrays
    credentials = re.compile(r"-----BEGIN .*PRIVATE KEY|\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b")
    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in {"qpos", "qvel", "actions", "raw_cem_action", "held_action", "password",
                    "private_key", "policy_weights", "compiler_command", "worker_pid"}
                inspect(item)
        elif isinstance(value, list):
            for item in value: inspect(item)
        elif isinstance(value, str):
            assert not value.startswith(("/", "~", "file:", "C:\\", "D:\\"))
            assert not any(path in value for path in ("/home/", "/data/", "/root/", "/tmp/", "/private/"))
            assert credentials.search(value) is None
    inspect(summary)


@pytest.mark.parametrize("complete", [False, True])
def test_asset_free_end_to_end_replay_publishes_complete_or_censored_grid_honestly(tmp_path, monkeypatch, complete):
    args = fixture_arguments(tmp_path, monkeypatch, complete=complete)
    originals = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    value = analysis.run(args)
    assert value["kind"] == "separate_sa29_mujoco_fixed_nmpc_friction_development"
    assert value["reference_id"] == REFERENCE_ID and value["status"] == "development_analysis_only"
    assert value["provided_mu"] == (GRID if complete else [.1, .3])
    assert value["evaluated_mu"] == (GRID if complete else [.1])
    assert value["censored_mu"] == ([] if complete else [.3])
    assert value["complete_predeclared_grid"] is complete
    assert value["missing_or_censored_mu"] == ([] if complete else [mu for mu in GRID if mu != .1])
    assert value["reused_nominal_pilot_verified"] is complete
    assert len(value["rows"]) == 17
    for name in ("pooled_success_statistics", "primary_task_replicate", "sim2sim_comparison", "isaac_evidence_available",
        "supplementary_policy_evidence_available", "two_engine_nominal_qualified", "qualified_freeze_created",
        "qualified_driver_executed", "frozen_benchmark", "eligible_for_controller_ranking", "robustness_certificate",
        "real_time_verified", "controlled_execution_success_claimed", "controller_retuned", "height_or_flight_failure_gate",
        "heading_offset_is_mismatch", "dynamics_stepping", "inverse_kinematics", "retargeting", "learned_inference",
        "new_policy_training_performed", "actuation_delay_injected"):
        assert value[name] is False
    arrays = analysis.load_arrays(args.output_dir / "development_trace.npz")
    assert_portable(value, arrays)
    assert ("mu0p300_time_s" in arrays) is complete
    assert set(path.name for path in args.output_dir.iterdir()) == {"summary.json", "development_trace.npz", "topdown.png"}
    for artifact in value["artifacts"].values():
        assert analysis.sha256(args.output_dir / artifact["file"]) == artifact["sha256"]
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_cross_case_runtime_contract_cannot_drift_in_an_otherwise_valid_partial_grid(tmp_path, monkeypatch):
    args = fixture_arguments(tmp_path, monkeypatch, complete=False)
    original = analysis.audit_episode
    def changed(*arguments, **keywords):
        evidence, trace, public = original(*arguments, **keywords)
        if public["actual_plant_mu"] == .3:
            public["controller_contract"] = {**public["controller_contract"], "synthetic_unexpected_revision": True}
        return evidence, trace, public
    monkeypatch.setattr(analysis, "audit_episode", changed)
    with pytest.raises(ValueError, match="across actual plant coefficients"):
        analysis.run(args)
    assert not args.output_dir.exists()


def test_all_supplied_cases_can_be_censored_without_inventing_any_physical_episode(tmp_path, monkeypatch):
    args = fixture_arguments(tmp_path, monkeypatch, complete=False)
    directory = tmp_path / analysis.case_name(.1)
    report = json.loads((directory / "report.json").read_text())
    report.update(status="running", completed_policy_steps=0)
    save_report(directory, report)
    def no_physical_evidence(*unused, **keywords):
        raise AssertionError("An incomplete report must not be promoted to physical evidence")
    monkeypatch.setattr(analysis, "episode_evidence", no_physical_evidence)
    value = analysis.run(args)
    assert value["evaluated_mu"] == [] and value["censored_mu"] == [.1, .3]
    assert value["missing_or_censored_mu"] == GRID and value["complete_predeclared_grid"] is False
    assert value["posture_diagnostics"] == value["contact_diagnostics"] == {}
    assert set(value["windows"]["core"]) == {"reference"}
    assert all(row["censored"] is True and row["NMPC"]["large_xy_exploratory_screen"] is None
        for row in value["rows"])
    arrays = analysis.load_arrays(args.output_dir / "development_trace.npz")
    assert not any(key.startswith("mu") for key in arrays)
    assert_portable(value, arrays)


@pytest.fixture(scope="module")
def published_sa29_grid():
    """Only three portable public files; no models, weights or raw states."""
    directory = ROOT / "results/nmpc_sa29_friction_development_20261016_v1"
    blobs = {path.name: path.read_bytes() for path in directory.iterdir()}
    with np.load(io.BytesIO(blobs["development_trace.npz"]), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return {"summary": json.loads(blobs["summary.json"]), "arrays": arrays,
        "hashes": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}}


def test_final_three_file_publication_pins_all_17_original_episodes_and_unchanged_sources(published_sa29_grid):
    summary = published_sa29_grid["summary"]; manifest = declaration_fixture()
    assert published_sa29_grid["hashes"] == PUBLIC_SHA256
    assert summary["kind"] == "separate_sa29_mujoco_fixed_nmpc_friction_development"
    assert summary["status"] == "development_analysis_only" and summary["engine"] == "MuJoCo"
    assert summary["reference_spec"] == get_reference(REFERENCE_ID)
    assert summary["reference_id"] == REFERENCE_ID and summary["role"] == "separate_supplement"
    assert summary["stratum"] == "high_sideslip_s_avoidance"
    assert summary["predeclaration_sha256"] == PREDECLARATION_SHA256
    assert summary["intended_local_episodes"] == 17 and summary["predeclared_new_local_episodes"] == 16
    assert summary["provided_mu"] == summary["evaluated_mu"] == summary["expected_mu"] == GRID
    assert summary["censored_mu"] == summary["missing_or_censored_mu"] == []
    assert summary["complete_predeclared_grid"] is summary["reused_nominal_pilot_verified"] is True
    assert summary["reused_nominal_input_sha256"] == RAW_IDENTITY_SHA256 == PUBLIC_RAW_IDENTITY_SHA256["mu0p700"]
    assert summary["frames_per_complete_episode"] == FRAMES
    assert summary["physical_samples_per_complete_episode"] == SAMPLES
    assert summary["duration_s_per_complete_episode"] == 5.16
    assert summary["physics_dt_s"] == .005 and summary["control_dt_s"] == .02
    assert summary["prediction_mu_scenarios"] == [.7] and summary["effective_cem_config"] == analysis.PILOT_CEM
    assert summary["effective_cem_config_sha256"] == manifest["effective_cem_configuration_sha256"]
    assert summary["input_sha256"] == {name: manifest["input_sha256"][name] for name in ("motion", "robot_reference", "model")}
    assert summary["reference_catalog_sha256"] == manifest["source_sha256"]["config/controller_references.json"]
    assert set(summary["runs"]) == set(PUBLIC_RAW_IDENTITY_SHA256)
    for mu in GRID:
        label = analysis.case_name(mu); run = summary["runs"][label]
        assert run["input_sha256"] == PUBLIC_RAW_IDENTITY_SHA256[label]
        assert run["actual_plant_mu"] == mu and run["plant_material_readback"]["compiled_sliding_friction_range"] == [mu, mu]
        assert run["prediction_mu_scenarios"] == [.7] and run["no_retuning"] is True
        assert run["complete_physical_evidence"] is True and run["censored"] is False
        assert run["nominal_pilot_reused"] is (mu == .7)
        assert run["runtime_source_sha256"] == manifest["source_sha256"]
        assert run["native_identity"]["native_library_sha256"] == manifest["input_sha256"]["native_library"]
        assert run["reported_settings"] == json.loads((ROOT / manifest["settings"]).read_text())
        provenance = {item["source_name"]: item["sha256"]
            for item in run["reported_source_provenance"]["reported_source_hashes"]}
        assert provenance == {**{Path(name).name: value for name, value in manifest["source_sha256"].items()},
            "policy_contract.json": manifest["input_sha256"]["actuator_contract"]}
    assert summary["analysis_source_sha256"] == {**analysis.FROZEN_HELPER_SHA256,
        "analyze_nmpc_sa29_friction_development.py": ANALYZER_SHA256}
    for name, digest in summary["analysis_source_sha256"].items():
        assert hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() == digest
    for item in summary["artifacts"].values():
        assert item["sha256"] == PUBLIC_SHA256[item["file"]]


def test_published_budget_and_common_score_retain_failed_xy_runs_without_a_solver_or_real_time_claim(published_sa29_grid):
    summary = published_sa29_grid["summary"]
    assert len(summary["objective_contract"]["weights"]) == 15
    assert summary["objective_contract"]["absolute_height_target"] is False
    assert summary["objective_contract"]["reference_vertical_velocity_target"] is False
    for run in summary["runs"].values():
        diagnostic = run["solver_diagnostics"]; counts = diagnostic["counts"]
        assert diagnostic["decision_samples"] == FRAMES
        assert diagnostic["common_score_terms"] == 15 and diagnostic["common_score_arithmetic_verified"] is True
        assert diagnostic["raw_float64_to_applied_float32_verified"] is True
        assert counts["candidate_evaluations"] == counts["finite_candidates"] == counts["scenario_evaluations"] == 50309
        assert counts["batch_prediction_calls"] == 1547 and counts["warm_start_used"] == 257
        assert all(counts[key] == 0 for key in ("failed_candidates", "failed_scenarios", "failed_batch_prediction_calls",
            "optimization_failed", "fallback_used", "initial_candidate_generation_failed"))
        assert run["simulation_budget"] == {"actual_plant_physics_substeps": 1032,
            "requested_prediction_substeps": 6037080, "failed_prediction_requests_prove_completed_steps": False}
        timing = diagnostic["timing"]
        assert timing["sample_count"] == FRAMES and timing["fraction_above_20ms"] == 1.
        assert timing["deadline_enforced"] is timing["real_time_verified"] is timing["actuation_delay_injected"] is False
    assert sum(run["simulation_budget"]["actual_plant_physics_substeps"] for run in summary["runs"].values()) == 17544
    assert sum(run["simulation_budget"]["requested_prediction_substeps"] for run in summary["runs"].values()) == 102630360


def test_all_published_clocks_geometry_core_events_contacts_speed_and_posture_replay_asset_free(published_sa29_grid):
    summary, arrays = published_sa29_grid["summary"], published_sa29_grid["arrays"]
    traces, recon = {}, {}
    for label in ("reference", *[analysis.case_name(mu) for mu in GRID]):
        # Reuses the frozen pilot's exact-clock and independent heading_signals
        # reconstruction check: FK is not called, and no model asset is loaded.
        traces[label], recon[label] = public_trace(arrays, label)
    metrics = analysis.summarize_cases(traces, recon, get_reference(REFERENCE_ID))
    assert metrics == summary["windows"]
    assert analysis.make_rows(metrics, summary["runs"]) == summary["rows"]
    for label, trace in traces.items():
        if label == "reference": continue
        assert posture_diagnostic(trace) == summary["posture_diagnostics"][label]
        assert contact_diagnostic(trace, [0., 3.14]) == summary["contact_diagnostics"][label]["core"]
        for event in summary["contact_diagnostics"][label]["supplementary_events"]:
            assert event["descriptive_not_gate"] is True
            assert contact_diagnostic(trace, event["interval_s"]) == event["diagnostic"]
    reference = traces["reference"]
    windows = [(metrics["core"], [0., 3.14], 629)] + [(event["metrics"], event["interval_s"], count)
        for event, count in zip(metrics["supplementary_events"], (153, 233))]
    for values, interval, count in windows:
        assert values["reference"]["ground_contact_heading_geometry"] is None
        assert values["reference"]["ground_contact_lateral_slip_m_s"] is None
        for label, trace in traces.items():
            if label == "reference": continue
            selected = (trace["time_s"] >= interval[0] - 1e-9) & (trace["time_s"] <= interval[1] + 1e-9)
            value = values[label]
            assert selected.sum() == value["samples_in_window"] == count
            target = np.column_stack([np.interp(trace["time_s"][selected], reference["time_s"],
                reference["position"][:, axis]) for axis in (0, 1)])
            error = np.linalg.norm(trace["position"][selected, :2] - target, axis=1)
            assert float(np.sqrt(np.mean(error ** 2))) == value["reference_xy_rmse_m"]
            assert float(error.max()) == value["reference_xy_peak_m"]
            assert value["large_xy_exploratory_screen"] is (value["reference_xy_rmse_m"] > .15 or value["reference_xy_peak_m"] > .3)
            assert value["heading_offset_is_mismatch"] is value["height_or_flight_failure_gate"] is False
            speed = np.linalg.norm(trace["com_velocity"][selected, :2], axis=1)
            assert float(speed.mean()) == value["mean_speed_m_s"]
            gate = (trace["contact_count"][selected] > 0) & (trace["contact_force"][selected, :, 2] > 5.)
            assert gate.mean(axis=0).tolist() == value["ground_contact_heading_geometry"]["contact_fraction_left_right"]
            for side, name in enumerate(("left", "right")):
                slip = trace["contact_slip"][selected, side][gate[:, side]]
                expected = float(np.sqrt(np.mean(slip ** 2))) if len(slip) else None
                assert expected == value["ground_contact_lateral_slip_m_s"][name]


def test_observed_xy_failures_remain_nonmonotonic_and_are_not_controlled_upright_success(published_sa29_grid):
    summary, arrays = published_sa29_grid["summary"], published_sa29_grid["arrays"]
    screened = [row["mu"] for row in summary["rows"] if row["NMPC"]["large_xy_exploratory_screen"]]
    assert screened == [.01, .03, .05, .1, .15, 1.4]
    rows = {row["mu"]: row["NMPC"] for row in summary["rows"]}
    assert rows[.7]["reference_xy_rmse_m"] == .10152020001087723
    assert rows[.7]["reference_xy_peak_m"] == .16018105988167902
    assert rows[.15]["mean_speed_retention_ratio"] == 1.3916299091170057
    assert rows[.15]["large_xy_exploratory_screen"] is True
    assert rows[.2]["reference_xy_rmse_m"] == .1174985973708917
    assert rows[.2]["reference_xy_peak_m"] == .24295571672913405
    assert rows[.2]["large_xy_exploratory_screen"] is False
    time, quaternion = arrays["mu0p200_time_s"], arrays["mu0p200_quaternion"]
    core = (time >= -1e-9) & (time <= 3.14 + 1e-9)
    assert posture_diagnostic({"quaternion": quaternion[core]})["max_base_tilt_deg"] > 90.
    # Display selection includes a known low-mu failure; it never filters the
    # complete 17-case data and is not a new runtime or success declaration.
    assert analysis.SELECTED_MU == (.01, .1, .7, 1.6, 2.)


def test_public_grid_projection_is_private_and_explicitly_not_policy_isaac_or_ranked_evidence(published_sa29_grid):
    summary, arrays = published_sa29_grid["summary"], published_sa29_grid["arrays"]
    assert_portable(summary, arrays)
    assert len({key.partition("_")[0] for key in arrays if key.startswith("mu")}) == 17
    assert summary["events_are_descriptive_not_success_gates"] is True
    for key in ("settings_changed_across_plant_mu", "pooled_success_statistics", "primary_task_replicate",
        "sim2sim_comparison", "isaac_evidence_available", "supplementary_policy_evidence_available",
        "two_engine_nominal_qualified", "qualified_freeze_created", "qualified_driver_executed",
        "frozen_benchmark", "eligible_for_controller_ranking", "robustness_certificate", "real_time_verified",
        "controlled_execution_success_claimed", "controller_retuned", "height_or_flight_failure_gate",
        "heading_offset_is_mismatch", "dynamics_stepping", "inverse_kinematics", "retargeting",
        "learned_inference", "new_policy_training_performed", "actuation_delay_injected"):
        assert summary[key] is False
