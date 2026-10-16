"""Asset-free replay of the fixed-CEM and recorded-policy friction bundles."""
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_controller_pair import pair_xy_gap, posture_diagnostic
from analyze_nmpc_nominal_development import contact_diagnostic
from analyze_policy_friction_dynamics import audit_xy_replay
from analyze_tvlqr_pilot import PUBLIC_SIGNALS, PUBLIC_TRACE_FIELDS
from controller_metrics import heading_signals, summarize_turn
from controller_references import get_reference


GRID = [.01, .03, .05, .1, .15, .2, .3, .45, .6, .7, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.]
BUNDLES = {
    "nmpc": ("nmpc_primary_mujoco_friction_development_20261016_v1", "development_trace.npz", {
        "summary.json": "36a0fabe1e061f97f95a06385e88c53411dd3da5754d27e46a693f686b7d7887",
        "development_trace.npz": "ef2eb62edd389d6c4d117060bb16b61203456792531685fd009a2908ad64277d",
        "topdown.png": "d4a72bf04c68871391cb2ebc845681d065cae883ad2ed533151165a7a945a718"}),
    "policy": ("policy_primary_friction_dynamics_20261016_v1", "policy_dynamics_trace.npz", {
        "summary.json": "476623073f062f0c285d763ffe6cfc32ca423bd666e0a40791852721ca48f1cb",
        "policy_dynamics_trace.npz": "1b757d06696ad6aea47d6c4f1be584005670d2b3274b82daa9a690a3e3e43a8d",
        "topdown.png": "36404698930a046e725dddfd5f5a1e75f9e29bec52d773514406c1343120d058"}),
}
CASE_INPUT_HASHES = {
    "nmpc": "2cb37b9aedf3846146c964bb23a46740f7bb65588067d3cad4b600990e3f251c",
    "policy": "760f96ded0c36919ecc6916292369c7245ccf120f0c39ca16e6753c78a94173c"}


def case(mu):
    return ("mu" + format(mu, ".3f")).replace(".", "p")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture(scope="module")
def published():
    """Load each large public archive once; never load a licensed/raw asset."""
    result = {}
    for key, (name, filename, expected) in BUNDLES.items():
        directory = ROOT / "results" / name
        blobs = {name: (directory / name).read_bytes() for name in expected}
        with np.load(io.BytesIO(blobs[filename]), allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        result[key] = {"summary": json.loads(blobs["summary.json"]), "arrays": arrays,
                       "hashes": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
                       "files": {path.name for path in directory.iterdir()}}
    result["legacy"] = {}
    for algorithm in ("PPO", "SAC"):
        directory = ROOT / "results" / ("controller_primary_" + algorithm.lower() + "_height_smooth_seed42_v1")
        with np.load(directory / "xy_trace.npz", allow_pickle=False) as archive:
            reference = {name: archive[name].copy() for name in ("reference_t", "reference_xy")}
        result["legacy"][algorithm] = {**reference,
            "summary": json.loads((directory / "summary.json").read_text()),
            "metadata": json.loads((directory / "training_metadata.json").read_text())}
    return result


def replay_arrays(arrays, prefix):
    trace = {name: arrays[prefix + "_" + name] for name in PUBLIC_TRACE_FIELDS if prefix + "_" + name in arrays}
    assert trace["time_s"].shape == ((349,) if prefix == "reference" else (1397,))
    interval = .02 if prefix == "reference" else .005
    np.testing.assert_allclose(trace["time_s"], np.arange(len(trace["time_s"])) * interval, atol=1e-9, rtol=0)
    assert float(trace["physics_dt_s"]) == interval
    signals = heading_signals(trace["quaternion"], trace["com_velocity"],
        arrays[prefix + "_wheel_quaternion_wxyz"], arrays[prefix + "_wheel_center_velocity_w"])
    for name in PUBLIC_SIGNALS:
        np.testing.assert_array_equal(signals[name], arrays[prefix + "_" + name])
    # An independent measured-COM speed check, not a position gradient.
    np.testing.assert_array_equal(signals["speed_m_s"],
        np.linalg.norm(np.asarray(trace["com_velocity"], dtype=float)[:, :2], axis=1))
    return trace, signals


def replay_window(trace, signals, reference, reference_signals):
    contact = {} if "contact_count" not in trace else {
        "wheel_ground_contact_count": trace["contact_count"], "wheel_ground_force_w": trace["contact_force"],
        "wheel_contact_lateral_slip_m_s": trace["contact_slip"]}
    return summarize_turn(trace["time_s"], trace["position"], signals, reference["time_s"],
        reference["position"][:, :2], reference_signals["speed_m_s"], [1.44, 4.], **contact)


def test_immutable_three_file_bundles_and_analysis_sources(published):
    for key, (_, _, expected) in BUNDLES.items():
        bundle = published[key]; summary = bundle["summary"]
        assert bundle["files"] == set(expected)
        assert bundle["hashes"] == expected
        assert summary["reference_spec"] == get_reference()
        assert summary["reference_catalog_sha256"] == "b2c6062ebc6c25d48d7845ffb8786971f85d7b44e37d5f6f27a9f65e395b96c1"
        for artifact in summary["artifacts"].values():
            assert artifact["sha256"] == expected[artifact["file"]]
        for name, sha in summary["analysis_source_sha256"].items():
            assert hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() == sha
        identities = {name: item["input_sha256"] for name, item in summary["runs"].items()}
        assert digest(identities) == CASE_INPUT_HASHES[key]
        assert all(re.fullmatch("[0-9a-f]{64}", sha) for row in identities.values() for sha in row.values())


def test_fixed_17_case_nmpc_identity_budget_and_unqualified_flags(published):
    summary = published["nmpc"]["summary"]
    assert summary["kind"] == "unqualified_mujoco_cem_fixed_friction_development"
    assert summary["expected_mu"] == summary["provided_mu"] == summary["evaluated_mu"] == GRID
    assert summary["censored_mu"] == summary["missing_or_censored_mu"] == []
    assert summary["complete_predeclared_grid"] is True
    for name in ("two_engine_nominal_qualified", "qualified_freeze_created", "qualified_driver_executed",
                 "sim2sim_comparison", "eligible_for_controller_ranking", "robustness_certificate",
                 "controlled_execution_success_claimed", "height_or_flight_failure_gate", "heading_offset_is_mismatch",
                 "dynamics_stepping", "learned_inference", "settings_changed_across_plant_mu"):
        assert summary[name] is False
    assert summary["effective_cem_config_sha256"] == "c30a83dffe39e167748d8305f02d43025dbb83af2d6539bd44b3de7ec4616a74"
    assert digest(summary["effective_cem_config"]) == summary["effective_cem_config_sha256"]
    assert summary["prediction_mu_scenarios"] == [.7]
    assert set(summary["runs"]) == {case(mu) for mu in GRID}
    plan = json.loads((ROOT / "config/nmpc_friction_development.json").read_text())
    for mu in GRID:
        run = summary["runs"][case(mu)]
        assert run["actual_plant_mu"] == mu and run["prediction_mu_scenarios"] == [.7]
        assert run["complete_physical_evidence"] is True and run["censored"] is False
        hashes = {item["source_name"]: item["sha256"] for item in run["reported_source_provenance"]["reported_source_hashes"]}
        for name, sha in plan["source_sha256"].items():
            assert hashes[Path(name).name] == sha
        assert run["native_identity"]["native_library_sha256"] == "b13e74df3a532496449d2f6b34b6934784a6bfd4ea5f0574c90cf279048821b7"
        counts = run["solver_diagnostics"]["counts"]
        assert counts["candidate_evaluations"] == counts["finite_candidates"] == counts["scenario_evaluations"] == 68054
        assert counts["failed_candidates"] == counts["failed_scenarios"] == counts["fallback_used"] == counts["optimization_failed"] == 0
        assert run["simulation_budget"]["requested_prediction_substeps"] == 8166480
        assert run["simulation_budget"]["actual_plant_physics_substeps"] == 1396
        assert run["solver_diagnostics"]["timing"]["real_time_verified"] is False


def test_nmpc_public_arrays_exactly_replay_xy_speed_heading_contact_and_posture(published):
    summary, arrays = published["nmpc"]["summary"], published["nmpc"]["arrays"]
    reference, signals = replay_arrays(arrays, "reference")
    assert replay_window(reference, signals, reference, signals) == summary["windows"]["core"]["reference"]
    failures = []
    for mu in GRID:
        name = case(mu); trace, measured = replay_arrays(arrays, name)
        result = replay_window(trace, measured, reference, signals)
        assert result == summary["windows"]["core"][name]
        assert posture_diagnostic(trace) == summary["posture_diagnostics"][name]
        assert contact_diagnostic(trace, [1.44, 4.]) == summary["contact_diagnostics"][name]
        if result["large_xy_exploratory_screen"]: failures.append(mu)
    assert failures == [.01, .03, .05, .1, .15, .2, .7]


def test_all_68_recorded_policy_episodes_exactly_replay_full_features(published):
    summary, arrays = published["policy"]["summary"], published["policy"]["arrays"]
    assert summary["kind"] == "existing_policy_friction_dynamics_replay" and summary["evaluation_mu"] == GRID
    reference, reference_signals = replay_arrays(arrays, "reference")
    assert len(summary["runs"]) == 34
    for algorithm in ("PPO", "SAC"):
        for mu in GRID:
            label = algorithm + "_" + case(mu); run = summary["runs"][label]
            assert run["algorithm"] == algorithm and run["mu"] == run["audit"]["fixed_mu"] == mu
            pair = {}
            for engine in ("isaac", "mujoco"):
                trace, signals = replay_arrays(arrays, label + "_" + engine); pair[engine] = trace
                assert replay_window(trace, signals, reference, reference_signals) == run["windows"]["core"][engine]
                assert posture_diagnostic(trace) == run["posture_diagnostics"][engine]
            assert pair_xy_gap(pair["isaac"], pair["mujoco"], [1.44, 4.]) == run["windows"]["core"]["sim2sim_xy_gap"]
    for name in ("new_policy_evaluation", "new_training", "dynamics_stepping", "learned_inference",
                 "eligible_for_controller_ranking", "controlled_execution_success_claimed", "height_or_flight_failure_gate",
                 "heading_offset_is_mismatch", "hardware_ready", "real_time_verified"):
        assert summary[name] is False


def test_policy_training_and_reference_quantization_bound_without_alignment(published):
    summary, arrays = published["policy"]["summary"], published["policy"]["arrays"]
    proof = summary["reference_numeric_representation_audit"]
    assert proof["kind"] == "verified_locked_reference_float64_to_float32_xy_export"
    assert proof["float32_cast_bitwise_equal"] is proof["identical_reference_timestamps"] is True
    assert proof["translation_rotation_or_time_warp_applied"] is proof["historical_reference_or_metrics_modified"] is False
    for algorithm in ("PPO", "SAC"):
        legacy = published["legacy"][algorithm]
        np.testing.assert_array_equal(arrays["reference_time_s"], legacy["reference_t"])
        original, exported = legacy["reference_xy"], arrays["reference_position"][:, :2]
        np.testing.assert_array_equal(exported, original.astype(np.float32).astype(np.float64))
        delta = float(np.linalg.norm(exported - original, axis=1).max())
        assert delta == proof["analytic_metric_change_bound_m"] == 2.395130692124446e-7
        metadata = legacy["metadata"]; source = summary["policies"][algorithm]["source_xy_grid"]
        assert source["training_budget"]["environment_transitions_completed"] == 12288000
        assert source["checkpoint_sha256"] == metadata["checkpoint_sha256"]
        original_rows = {row["evaluation_mu_static_dynamic_equal"]: row for row in legacy["summary"]["rows"]}
        for mu in GRID:
            label = algorithm + "_" + case(mu); run = summary["runs"][label]
            assert run["training"]["verified_formal_training"] is True
            assert run["training"]["manifest_sha256"] == metadata["training_manifest_sha256"]
            assert run["training"]["checkpoint_sha256"] == metadata["checkpoint_sha256"]
            for engine in ("isaac", "mujoco"):
                xy = arrays[label + "_" + engine + "_position"][:, :2]
                replay = audit_xy_replay(run["windows"]["core"][engine], original_rows[mu][engine],
                    reference_audit=proof, observed_xy=xy)
                assert replay == run["xy_numeric_representation_replay"][engine]
                assert replay["xy_screen_unchanged"] is True


def test_public_projection_has_no_raw_states_actions_paths_or_credentials(published):
    public_fields = set(PUBLIC_TRACE_FIELDS) | set(PUBLIC_SIGNALS) | {"wheel_quaternion_wxyz", "wheel_center_velocity_w"}
    private = re.compile(r"-----BEGIN .*PRIVATE KEY|\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b")

    def inspect_json(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in {"qpos", "qvel", "actions", "raw_cem_action", "policy_weights", "password", "private_key"}
                inspect_json(item)
        elif isinstance(value, list):
            for item in value: inspect_json(item)
        elif isinstance(value, str):
            assert not value.startswith(("/", "~", "file:", "C:\\", "D:\\"))
            assert not any(path in value for path in ("/home/", "/data/", "/root/", "/tmp/"))
            assert not private.search(value)

    for key in BUNDLES:
        bundle = published[key]; inspect_json(bundle["summary"])
        prefixes = ["reference"] + ([case(mu) for mu in GRID] if key == "nmpc" else
            [algorithm + "_" + case(mu) + "_" + engine for algorithm in ("PPO", "SAC") for mu in GRID for engine in ("isaac", "mujoco")])
        allowed = {"schema_version", "wheel_order"} | {prefix + "_" + field for prefix in prefixes for field in public_fields}
        if key == "nmpc":
            allowed |= {algorithm + "_" + case(mu) + "_" + field for algorithm in ("ppo", "sac") for mu in GRID for field in ("time_s", "xy")}
        assert set(bundle["arrays"]) <= allowed
        assert len(bundle["arrays"]) == (551 if key == "nmpc" else 1860)
        for name, values in bundle["arrays"].items():
            assert not values.dtype.hasobject
            if values.dtype.kind in "biufc": assert np.isfinite(values).all()
            else:
                for value in values.flat: inspect_json(str(value))
