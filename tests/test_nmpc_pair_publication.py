"""Asset-free replay of the FINAL 17-case paired friction publication.

Only derived public arrays/JSON are read. No simulator, licensed robot asset,
policy checkpoint, inference, raw controller state or network is needed.
"""
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pytest

from test_controller_friction_publication import (
    BUNDLES, GRID, ROOT, case, digest, replay_arrays, replay_window,
)
from analyze_controller_pair import pair_xy_gap, posture_diagnostic
from analyze_nmpc_nominal_development import contact_diagnostic
from analyze_tvlqr_pilot import PUBLIC_SIGNALS, PUBLIC_TRACE_FIELDS
from controller_references import get_reference


DIRECTORY = ROOT/"results/nmpc_primary_friction_pair_development_20261016_v1"
# Filled only after the complete final publication is produced and reviewed.
# A present bundle with missing pins MUST fail, not silently become a skip.
EXPECTED_HASHES = {
    "summary.json": "f157e7894890560f330c3d3024cac9d4810c6db852f57b5dfe0939d5e5465191",
    "development_trace.npz": "2bfcc0c1c16eb57a8238e85d8d12a6392d02fe4e6c3b213ef57e0f67a4745ac5",
    "topdown.png": "296527aca6a5482a93e76b6b30bfb983b8aee2da0da8eea55201c97ad63638a4",
}
EXPECTED_ISAAC_INPUTS_SHA256 = "25d7720bd6e871d773f89c8d5dd09ca67c6dc98850b3335d440b5be8cf878afc"


@pytest.fixture(scope="module")
def paired_publication():
    if not DIRECTORY.is_dir():
        pytest.skip("Final paired publication has not been generated yet")
    assert EXPECTED_HASHES and EXPECTED_ISAAC_INPUTS_SHA256, "Final bundle requires reviewed immutable publication pins"
    assert {path.name for path in DIRECTORY.iterdir()} == {"summary.json", "development_trace.npz", "topdown.png"}
    blobs = {name: (DIRECTORY/name).read_bytes() for name in EXPECTED_HASHES}
    with np.load(io.BytesIO(blobs["development_trace.npz"]), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    sources = {}
    for key, (directory, filename, expected) in BUNDLES.items():
        folder = ROOT/"results"/directory
        hashes = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in expected}
        assert hashes == expected
        sources[key] = {"summary": json.loads((folder/"summary.json").read_text()), "hashes": hashes}
        if key == "policy":
            with np.load(folder/filename, allow_pickle=False) as archive:
                sources[key]["array_names"] = set(archive.files)
    return {"summary": json.loads(blobs["summary.json"]), "arrays": arrays,
        "hashes": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}, "sources": sources}


def test_final_three_file_hashes_source_bundles_and_complete_unqualified_scope(paired_publication):
    bundle = paired_publication; summary = bundle["summary"]
    assert bundle["hashes"] == EXPECTED_HASHES
    assert summary["kind"] == "unqualified_two_engine_cem_friction_development"
    assert summary["reference_spec"] == get_reference()
    assert summary["expected_mu"] == summary["provided_isaac_mu"] == summary["evaluated_isaac_mu"] == GRID
    assert summary["missing_or_censored_isaac_mu"] == [] and summary["complete_two_engine_grid"] is True
    assert summary["sim2sim_trajectory_comparison"] is True
    assert summary["policy_feature_scope"] == "original_physical_recording_dynamics"
    for name in ("two_engine_nominal_qualified", "qualified_freeze_created", "qualified_driver_executed",
        "frozen_benchmark", "pooled_success_statistics", "eligible_for_controller_ranking", "robustness_certificate",
        "real_time_verified", "controlled_execution_success_claimed", "height_or_flight_failure_gate",
        "heading_offset_is_mismatch", "dynamics_stepping", "learned_inference", "actuation_delay_injected"):
        assert summary[name] is False
    assert summary["mujoco_bundle_sha256"] == bundle["sources"]["nmpc"]["hashes"]
    assert summary["policy_dynamics_bundle"]["input_sha256"] == bundle["sources"]["policy"]["hashes"]
    assert summary["policy_dynamics_bundle"]["original_physical_archives_used"] is True
    for artifact in summary["artifacts"].values(): assert artifact["sha256"] == EXPECTED_HASHES[artifact["file"]]
    for name, value in summary["analysis_source_sha256"].items():
        assert Path(name).name == name
        assert hashlib.sha256((ROOT/"scripts"/name).read_bytes()).hexdigest() == value


def test_all17_isaac_material_device_fixed_controller_and_real_solver_counts(paired_publication):
    summary = paired_publication["summary"]
    assert len(summary["rows"]) == len(summary["isaac_runs"]) == 17
    assert [row["mu"] for row in summary["rows"]] == GRID
    assert summary["effective_cem_config_sha256"] == digest(summary["effective_cem_config"]) == "c30a83dffe39e167748d8305f02d43025dbb83af2d6539bd44b3de7ec4616a74"
    assert summary["prediction_mu_scenarios"] == [.7]
    identities = {name: run["input_sha256"] for name, run in summary["isaac_runs"].items()}
    assert digest(identities) == EXPECTED_ISAAC_INPUTS_SHA256
    assert all(re.fullmatch("[0-9a-f]{64}", value) for row in identities.values() for value in row.values())
    assets = set()
    for mu in GRID:
        run = summary["isaac_runs"][case(mu)]; assets.add(run["asset_sha256"])
        assert run["actual_plant_mu"] == mu and run["plant_seed"] == 42
        assert run["complete_physical_evidence"] is run["actual_controller_identity_available"] is True
        assert run["censored"] is False
        material = run["material_audit"]
        np.testing.assert_allclose(material["robot_static_dynamic_range"], mu, atol=1e-6, rtol=0)
        assert material["expected_mu"] == mu and material["ground_bound_materials"]
        for ground in material["ground_bound_materials"]:
            np.testing.assert_allclose(ground["coefficients"], [mu, mu, 0.], atol=1e-6, rtol=0)
        device = run["device_audit"]
        assert device["status"] == "verified" and device["native_and_tensor_device_consistency_verified"] is True
        assert device["cuda_visible_devices"] is None and device["required_unmasked"] is True
        assert run["native_identity"]["native_library_sha256"] == "ce676e8414d38beb0676e8529cd88ebfea980ef9d91494c052e21414c4f89ea5"
        counts = run["solver_diagnostics"]["counts"]
        assert counts["finite_candidates"]+counts["failed_candidates"] == counts["candidate_evaluations"] == counts["scenario_evaluations"]
        assert 0 <= counts["failed_scenarios"] <= counts["scenario_evaluations"]
        assert 0 <= counts["fallback_used"] == counts["optimization_failed"] <= 349
        assert run["simulation_budget"]["requested_prediction_substeps"] == counts["scenario_evaluations"]*30*4
        assert run["simulation_budget"]["actual_plant_physics_substeps"] == 1396
        assert run["solver_diagnostics"]["timing"]["real_time_verified"] is False
    assert len(assets) == 1


def test_all34_nmpc_engine_episodes_replay_xy_speed_headings_contact_posture_and_gaps(paired_publication):
    summary, arrays = paired_publication["summary"], paired_publication["arrays"]
    reference, reference_signals = replay_arrays(arrays, "reference")
    mj = paired_publication["sources"]["nmpc"]["summary"]
    for row in summary["rows"]:
        label = case(row["mu"]); pair = {}
        for engine in ("isaac", "mujoco"):
            prefix = engine+"_"+label; trace, signals = replay_arrays(arrays, prefix); pair[engine] = trace
            assert replay_window(trace, signals, reference, reference_signals) == row["NMPC"][engine]
            if engine == "isaac":
                assert posture_diagnostic(trace) == summary["isaac_posture_diagnostics"][prefix]
                assert contact_diagnostic(trace, [1.44, 4.]) == summary["isaac_contact_diagnostics"][prefix]
            else:
                assert row["NMPC"][engine] == mj["windows"]["core"][label]
                assert posture_diagnostic(trace) == mj["posture_diagnostics"][label]
                assert contact_diagnostic(trace, [1.44, 4.]) == mj["contact_diagnostics"][label]
        assert pair_xy_gap(pair["isaac"], pair["mujoco"], [1.44, 4.]) == row["NMPC"]["sim2sim"]


def test_all68_original_policy_engine_episodes_replay_full_features_and_unchanged_source(paired_publication):
    summary, arrays = paired_publication["summary"], paired_publication["arrays"]
    reference, reference_signals = replay_arrays(arrays, "reference")
    policy = paired_publication["sources"]["policy"]["summary"]
    assert summary["policy_dynamics_bundle"]["reference_numeric_representation_audit"] == policy["reference_numeric_representation_audit"]
    for row in summary["rows"]:
        label = case(row["mu"])
        for algorithm in ("PPO", "SAC"):
            original = policy["runs"][algorithm+"_"+label]; pair = {}
            for engine in ("isaac", "mujoco"):
                prefix = algorithm.lower()+"_"+label+"_"+engine
                trace, signals = replay_arrays(arrays, prefix); pair[engine] = trace
                result = replay_window(trace, signals, reference, reference_signals)
                assert result == row[algorithm][engine] == original["windows"]["core"][engine]
                assert posture_diagnostic(trace) == row[algorithm]["posture_diagnostics"][engine] == original["posture_diagnostics"][engine]
                np.testing.assert_array_equal(trace["position"][:, :2], arrays[prefix+"_xy"])
                for field in ("nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n"):
                    source_key = algorithm+"_"+label+"_"+engine+"_"+field
                    assert (prefix+"_"+field in arrays) == (source_key in paired_publication["sources"]["policy"]["array_names"])
            assert pair_xy_gap(pair["isaac"], pair["mujoco"], [1.44, 4.]) == row[algorithm]["sim2sim"]


def test_final_compact_projection_is_allowlisted_finite_and_private_data_free(paired_publication):
    arrays, summary = paired_publication["arrays"], paired_publication["summary"]
    assert "reference_contact_count" not in arrays and "reference_contact_force" not in arrays
    fields = set(PUBLIC_TRACE_FIELDS) | set(PUBLIC_SIGNALS) | {"wheel_quaternion_wxyz", "wheel_center_velocity_w"}
    prefixes = ["reference"]+[engine+"_"+case(mu) for mu in GRID for engine in ("isaac", "mujoco")]
    policy_prefixes = [algorithm+"_"+case(mu)+"_"+engine for algorithm in ("ppo", "sac") for mu in GRID for engine in ("isaac", "mujoco")]
    allowed = {"schema_version", "wheel_order"} | {prefix+"_"+field for prefix in prefixes+policy_prefixes for field in fields}
    allowed |= {prefix+"_xy" for prefix in policy_prefixes}
    assert set(arrays) <= allowed
    for array in arrays.values():
        assert not array.dtype.hasobject
        if array.dtype.kind in "biufc": assert np.isfinite(array).all()
    payload = json.dumps(summary)
    assert not any(token in payload for token in ("/home/", "/data/", "/root/", "/tmp/", "/private/", "compiler_command", "worker_pid"))
    assert not re.search(r"-----BEGIN .*PRIVATE KEY|\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b", payload)
    forbidden = {"qpos", "qvel", "actions", "raw_cem_action", "packed_controller_states", "policy_weights", "password", "private_key"}
    def inspect(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values(): inspect(item)
        elif isinstance(value, list):
            for item in value: inspect(item)
        elif isinstance(value, str):
            assert not value.startswith(("/", "~", "file:", "C:\\", "D:\\"))
    inspect(summary)
