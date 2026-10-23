"""Recover saved V1 force measurements after its EXPLICIT finalization failure.

Never fabricates a completed report, original-A validation proof, unavailable
mass/engine/USD readbacks, or new simulation. All input files remain untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import eval_ppo_ground_force_api_probe_v2 as probe
from ppo_backend_400hz_contract import read_json, require, sha256, verify_recording, write_json

V1_SHA = "3c36b568592d9d3f4d242aa271f0e6d772365df15c8d09ac00e6c1e561761deb"
STATUS = "FAILED_FINALIZATION_ARRAYS_RECOVERED"


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def validate_arrays(arrays, raw):
    n = probe.SAMPLES
    for name in ("counter", "sim_index", "sim_time", "time_s"):
        value = np.asarray(arrays[name])
        require(value.shape == (n,) and np.isfinite(value).all(), "Incomplete/nonfinite probe clock: "+name)
    for name in ("counter", "sim_index"):
        require(np.array_equal(arrays[name], np.round(arrays[name])) and (np.diff(arrays[name]) == 1).all(),
                "Probe physical counters are not consecutive integers")
    require(np.array_equal(arrays["counter"], np.asarray(raw["simulation_step_counter"])[1:])
            and np.array_equal(arrays["time_s"], np.asarray(raw["time_s"])[1:])
            and np.allclose(np.diff(arrays["sim_time"]), .0025, rtol=0, atol=2e-7),
            "Probe clock does not exactly bind native recorder[1:]")
    vector_names = ("point_sum_w_n", "legacy32_direct_matrix_w_n", "filtered64_matrix_w_n", "legacy_recorder_force_w_n")
    for name in vector_names:
        value = np.asarray(arrays[name])
        require(value.shape == (n, 2, 3) and np.isfinite(value).all(), "Incomplete/nonfinite force channel: "+name)
    for name in ("point_counts", "point_starts", "point_buffer_lengths", "point_counts_gt5N", "legacy32_capacity", "legacy_recorder_count_gt5N"):
        value = np.asarray(arrays[name])
        require(value.shape == (n, 2) and np.isfinite(value).all()
                and np.array_equal(value, np.round(value)), "Invalid raw count/start/capacity: "+name)
    counts, starts, lengths = arrays["point_counts"], arrays["point_starts"], arrays["point_buffer_lengths"]
    require((arrays["legacy32_capacity"] == 32).all() and (lengths == 32).all()
            and (counts >= 0).all() and (counts <= lengths).all()
            and ((counts == 0) | ((starts >= 0) & (starts+counts <= lengths))).all()
            and (arrays["point_counts_gt5N"] >= 0).all() and (arrays["point_counts_gt5N"] <= counts).all(),
            "Raw contact occupancy/start/capacity invalid")
    require(np.array_equal(arrays["legacy_recorder_force_w_n"], raw["wheel_ground_contact_force_w_n"][1:, 0])
            and np.array_equal(arrays["legacy_recorder_count_gt5N"], raw["wheel_ground_contact_count"][1:, 0]),
            "Saved recorder channels differ from native physics recording")
    pairs = (("point_sum_vs_direct32", "point_sum_w_n", "legacy32_direct_matrix_w_n"),
        ("point_sum_vs_legacy_recorder", "point_sum_w_n", "legacy_recorder_force_w_n"),
        ("direct32_vs_filtered64", "legacy32_direct_matrix_w_n", "filtered64_matrix_w_n"),
        ("point_sum_vs_filtered64", "point_sum_w_n", "filtered64_matrix_w_n"))
    return {"samples": n, "native_recorder_states": 2793, "probe_samples_bind_native_indices": [1, 2792],
        "comparisons": {label: probe.force_comparison(arrays[first], arrays[second]) for label, first, second in pairs},
        "legacy32_max_occupied_points_L_R": counts.max(0).tolist(), "legacy32_capacity": [32, 32],
        "legacy32_at_capacity_samples_L_R": np.count_nonzero(counts >= lengths, axis=0).tolist(),
        "legacy32_possible_saturation": bool(np.any(counts >= lengths)),
        "point_count_gt5N_vs_legacy_recorder_mismatches": np.count_nonzero(
            arrays["point_counts_gt5N"] != arrays["legacy_recorder_count_gt5N"], axis=0).tolist(),
        "force_units": "N; all-point normal sum retained, no force normalization or pointcount support gate"}


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Fresh recovery directory required; never overwrite failed evidence")
    require(sha256(probe.__file__) == V1_SHA, "Frozen failed V1 source changed")
    inputs = {name: Path(getattr(args, name)) for name in ("raw", "probe_arrays", "source_audit", "failure_log")}
    require(all(path.is_file() and not path.is_symlink() for path in inputs.values()), "Missing/indirect recovery input")
    before = {name: sha256(path) for name, path in inputs.items()}
    audit = read_json(inputs["source_audit"])
    require(audit.get("wrapper_sha256") == V1_SHA and audit.get("kind") == probe.base.KIND
            and audit.get("normalized_actor_sha256") == probe.ACTOR_SHA
            and audit.get("checkpoint_sha256") == probe.CHECKPOINT_SHA
            and audit.get("recording_sha256") == before["raw"] and audit.get("native_states") == 2793
            and audit.get("policy_decisions") == 349 and audit.get("validation_backend") == "PhysX",
            "Base-finish source audit does not bind this failed V1 A-actor recording")
    failure = inputs["failure_log"].read_text(errors="replace")
    require("FileExistsError" in failure and "evaluation_source_audit.json" in failure,
            "Missing explicit exclusive-write finalization failure evidence")
    recording = verify_recording(inputs["raw"], "PhysX")
    diagnostics = validate_arrays(load_npz(inputs["probe_arrays"]), load_npz(inputs["raw"]))
    require(all(sha256(path) == before[name] for name, path in inputs.items()), "Recovery input changed during read-only audit")
    result = {"status": STATUS, "kind": "failed_V1_saved_force_measurement_recovery",
        "role": probe.ROLE, "original_A_formal_result": False, "finalization_completed": False,
        "new_physics_inference_training_or_retry": False, "goal_completion_claimed": False,
        "actor_sha256_from_pre_failure_source_audit": probe.ACTOR_SHA,
        "checkpoint_sha256_from_pre_failure_source_audit": probe.CHECKPOINT_SHA,
        "raw_recording_verified": recording, "diagnostics": diagnostics,
        "input_files": {name: {"path": str(path.resolve()), "sha256": before[name], "bytes": path.stat().st_size}
            for name, path in inputs.items()},
        "source_sha256": {str(Path(__file__).resolve()): sha256(__file__), str(Path(probe.__file__).resolve()): V1_SHA},
        "unavailable_not_fabricated": ["probe_audit.json", "final report.json", "live per-body mass/gravity probe metadata", "optional USD gravity readback"],
        "limitations": ["V1 failed after writing complete native/probe arrays; recovery does not repair or invent its missing final report.",
            "The pre-failure A source audit is input evidence only; this recovery is NOT an original-A formal result.",
            "Raw aggregate force channels are observed values; individual point force buffers were not serialized.",
            "Capacity occupancy is a possible truncation flag, not a physical flight label."]}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"recovery_audit.json", result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("raw", "probe-arrays", "source-audit", "failure-log", "output-dir"):
        result.add_argument("--"+name, type=Path, required=True)
    return result


if __name__ == "__main__": run(parser().parse_args())
