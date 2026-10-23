"""Measurement-only PhysX contact-API probe of the SAME completed A actor.

Not a replacement for A's formal episode. Adds two filtered64 contact sensors
and observes scene.update only; no reward owner, actor/physics/state override.
The independent kind and wrapper SHA deliberately fail A's strict result gate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

import eval_ppo_ground_contact_v2 as base
import eval_ppo_backend_400hz_isaac as physx
import ppo_ground_contact_v2 as collector

require, sha256, write_json = base.require, base.sha256, base.write_json
KIND = "ground_contact_v2_A_actor_force_API_measurement_probe"
ROLE = "measurement_only_not_original_A_formal_comparison"
BASE_SHA = "86a2edf95961ddad29154cdc3aacb4f7c43a177a08b1e14d56bf03792d7ed578"
COLLECTOR_SHA = "ab0ddac6883b5e3639b357978116af705320314714517d5afa34a76b3d2b1fc5"
ACTOR_SHA = "0e489750f0f7377c698cd38031d68b104f859fe0b2e6a7b23bec452bf746ac39"
CHECKPOINT_SHA = "ee15d6856392dd541988f68324fba30d1c144db758c0ca8ce17f9ee4daac6562"
LEGACY_SENSORS = ("wheel_L_ground", "wheel_R_ground")
DT, SAMPLES = .0025, 2792


def numpy(value):
    if hasattr(value, "detach"): value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def matrix_vector(value):
    value = numpy(value)
    require(value.shape in ((1, 1, 3), (1, 1, 1, 3)) and np.isfinite(value).all(),
            "Require one environment/body/floor normal-force matrix")
    return value.reshape(3)


def point_sum(buffers):
    """Sum EVERY reported point;5N applies only to diagnostic count/masks."""
    require(len(buffers) == 6, "Unexpected get_contact_data tuple")
    force, _, normals, _, counts, starts = map(numpy, buffers)
    force, normals = force.reshape(-1), normals.reshape(-1, 3)
    require(counts.shape == starts.shape and counts.size == 1 and len(force) == len(normals), "Unexpected raw buffer layout")
    n, start = float(counts.item()), float(starts.item())
    require(n.is_integer() and start.is_integer() and n >= 0, "Invalid count/start integer")
    n, start = int(n), int(start)
    require(not n or (start >= 0 and start+n <= len(force)), "Raw contact buffer overrun")
    selected_force, selected_normal = force[start:start+n] if n else force[:0], normals[start:start+n] if n else normals[:0]
    lengths = np.linalg.norm(selected_normal, axis=-1)
    require(np.isfinite(selected_force).all() and (selected_force >= 0).all()
            and np.isfinite(selected_normal).all() and (lengths > 1e-12).all(), "Invalid active contact forces/normals")
    total = np.sum(selected_force[:, None]*selected_normal/lengths[:, None], axis=0)
    return total, n, start, len(force), int(np.count_nonzero(selected_force > 5.))


def force_comparison(first, second):
    first, second = np.asarray(first), np.asarray(second)
    require(first.shape == second.shape == (SAMPLES, 2, 3) and np.isfinite(first).all()
            and np.isfinite(second).all(), "Incomplete/nonfinite compared force arrays")
    a, b = first[..., 2] > 5., second[..., 2] > 5.
    return {"max_abs_vector_difference_n": float(np.max(np.abs(first-second))),
        "rms_vector_difference_n": float(np.sqrt(np.mean((first-second)**2))),
        "per_wheel_Fz_gt5N_mask_disagreement_samples": np.count_nonzero(a != b, axis=0).tolist(),
        "any_wheel_Fz_gt5N_mask_disagreement_samples": int(np.count_nonzero(a.any(-1) != b.any(-1))),
        "threshold_n_strict": 5., "pointcount_gate": False}


def mass_gravity(env):
    robot = env.scene["robot"]
    masses = numpy(robot.root_physx_view.get_masses())
    gravity = np.asarray(env.cfg.sim.gravity, dtype=float)
    require(masses.shape == (1, len(robot.body_names)) and np.isfinite(masses).all() and (masses >= 0).all()
            and masses.sum() > 0 and gravity.shape == (3,) and np.isfinite(gravity).all()
            and gravity[0] == gravity[1] == 0 and gravity[2] < 0, "Invalid live mass/configured world gravity")
    return {"body_names": list(robot.body_names), "body_mass_kg": masses[0].tolist(), "total_mass_kg": float(masses.sum()),
        "mass_source": "live robot.root_physx_view.get_masses()", "gravity_vector_m_s2": gravity.tolist(),
        "gravity_source": "env.cfg.sim.gravity (explicit configured source, not inferred USD readback)",
        "mg_n_for_diagnostic_only": float(masses.sum()*np.linalg.norm(gravity)), "force_signals_normalized": False}


def usd_gravity():
    try:
        import omni.usd
        from pxr import UsdPhysics
        rows = []
        for prim in omni.usd.get_context().get_stage().Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene = UsdPhysics.Scene(prim)
                rows.append({"prim_path": str(prim.GetPath()),
                    "gravity_direction": list(scene.GetGravityDirectionAttr().Get()),
                    "gravity_magnitude": float(scene.GetGravityMagnitudeAttr().Get())})
        return {"available": bool(rows), "source": "USD UsdPhysics.Scene authored attributes; not substitute for configured gravity", "scenes": rows}
    except Exception as error:
        return {"available": False, "reason": type(error).__name__+": "+str(error)}


class ForceAPIObserver:
    def __init__(self, env):
        require(env.num_envs == 1, "Probe requires one environment")
        self.env, self.previous = env, env.scene.update
        self.last_clock = collector.clock(env)
        self.start_clock = self.last_clock
        self.rows = []
        self.mass = mass_gravity(env)
        for side, (legacy, new) in enumerate(zip(LEGACY_SENSORS, collector.SENSORS)):
            for name, capacity in ((legacy, 32), (new, 64)):
                sensor = env.scene.sensors[name]
                require(tuple(sensor.body_names) == (collector.WHEELS[side],)
                    and tuple(sensor.cfg.filter_prim_paths_expr) == (collector.GROUND,)
                    and sensor.cfg.max_contact_data_count_per_prim == capacity
                    and sensor.contact_physx_view.filter_count == 1, "Sensor identity/filter/capacity differs")
        def update(*args, **kwargs):
            result = self.previous(*args, **kwargs)
            self.capture(args[0] if args else kwargs.get("dt"))
            return result
        self.update = update
        env.scene.update = update

    def capture(self, dt):
        current = collector.clock(self.env)
        if current[0] == self.last_clock[0]: return  # Initial/forward/reset refresh is not a physics sample.
        require(current[0] == self.last_clock[0]+1 and current[1] == self.last_clock[1]+1
                and len(self.rows) < SAMPLES and math.isclose(float(dt), DT, rel_tol=0, abs_tol=1e-12)
                and abs(current[2]-self.last_clock[2]-DT) < 2e-7, "Missing/duplicate/nonphysical observer clock")
        points, matrices, new_matrices, counts, starts, lengths, active_counts, capacities = [], [], [], [], [], [], [], []
        for legacy, new in zip(LEGACY_SENSORS, collector.SENSORS):
            old_sensor = self.env.scene.sensors[legacy]
            view = old_sensor.contact_physx_view
            matrices.append(matrix_vector(view.get_contact_force_matrix(dt=DT)))
            point, count, start, length, active = point_sum(view.get_contact_data(dt=DT))
            points.append(point); counts.append(count); starts.append(start); lengths.append(length); active_counts.append(active)
            capacity = int(view.max_contact_data_count)
            require(capacity == 32 and length == capacity and count <= capacity, "Legacy32 native contact capacity differs")
            capacities.append(capacity)
            new_matrices.append(matrix_vector(self.env.scene.sensors[new].data.force_matrix_w))
        self.rows.append({"counter": current[0], "sim_index": current[1], "sim_time": current[2],
            "point_sum_w_n": points, "legacy32_direct_matrix_w_n": matrices, "filtered64_matrix_w_n": new_matrices,
            "point_counts": counts, "point_starts": starts, "point_buffer_lengths": lengths,
            "point_counts_gt5N": active_counts, "legacy32_capacity": capacities})
        self.last_clock = current

    def finish(self, raw):
        require(len(self.rows) == SAMPLES and self.last_clock[0]-self.start_clock[0] == SAMPLES,
                "Probe requires all2792 actual physics updates")
        time = np.asarray(raw["time_s"])
        require(time.shape == (2793,) and np.allclose(time, np.arange(2793)*DT, atol=1e-10, rtol=0), "Native recorder clock differs")
        arrays = {name: np.asarray([row[name] for row in self.rows]) for name in self.rows[0]}
        arrays["time_s"] = time[1:].copy()
        arrays["legacy_recorder_force_w_n"] = np.asarray(raw["wheel_ground_contact_force_w_n"])[1:, 0].copy()
        arrays["legacy_recorder_count_gt5N"] = np.asarray(raw["wheel_ground_contact_count"])[1:, 0].copy()
        counts, capacity = arrays["point_counts"], arrays["legacy32_capacity"]
        pairs = (("point_sum_vs_direct32", "point_sum_w_n", "legacy32_direct_matrix_w_n"),
                 ("point_sum_vs_legacy_recorder", "point_sum_w_n", "legacy_recorder_force_w_n"),
                 ("direct32_vs_filtered64", "legacy32_direct_matrix_w_n", "filtered64_matrix_w_n"),
                 ("point_sum_vs_filtered64", "point_sum_w_n", "filtered64_matrix_w_n"))
        return arrays, {"samples": SAMPLES, "initial_same_counter_samples_excluded": True,
            "native_clock_alignment": "probe sample i maps exactly to physics_trajectory native state i+1",
            "clock_start": list(self.start_clock), "clock_end": list(self.last_clock),
            "legacy32_max_occupied_points_L_R": counts.max(0).tolist(), "legacy32_capacity": capacity[0].tolist(),
            "legacy32_at_capacity_samples_L_R": np.count_nonzero(counts >= capacity, axis=0).tolist(),
            "legacy32_possible_saturation": bool(np.any(counts >= capacity)),
            "saturation_interpretation": "Occupancy reaching capacity flags possible truncation, not proof of contact loss",
            "point_count_gt5N_vs_legacy_recorder_mismatches": np.count_nonzero(arrays["point_counts_gt5N"] != arrays["legacy_recorder_count_gt5N"], axis=0).tolist(),
            "comparisons": {label: force_comparison(arrays[first], arrays[second]) for label, first, second in pairs},
            "mass_and_gravity": self.mass}


def prepare_plan(argv):
    require(sha256(base.__file__) == BASE_SHA and sha256(collector.__file__) == COLLECTOR_SHA
            and sha256(physx.__file__) == base.PHYSX_HELPER_SHA256, "Pinned A evaluator/collector/helper changed")
    args = base.parser().parse_args(argv)
    require(args.backend == "physx" and args.mu == .7 and not args.allow_smoke_policy,
            "Probe requires nominal full-horizon PhysX and the completed formal A actor")
    plan = base.prepare_plan(argv)
    require(plan["cohort"]["input_sha256"]["policy.pt"] == ACTOR_SHA
            and plan["cohort"]["input_sha256"]["model_final.pt"] == CHECKPOINT_SHA,
            "Probe must reuse SAME completed A actor/checkpoint")
    plan["wrapper_sha256"] = sha256(__file__)
    return plan


def run(argv):
    plan = prepare_plan(argv)
    generated = physx.materialize_source(plan["evaluator"].read_text(), plan["legacy_root"])
    plan["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
    def configure(cfg, profile, plan):
        physx.configure(cfg, profile, plan)
        collector.configure_sensors(cfg.scene)  # Sensor config ONLY, never collector.install/reward_term.
    def audit_materials(env, plan):
        result = physx.audit_materials(env, plan)
        plan["observer"] = ForceAPIObserver(env)
        plan["usd_gravity"] = usd_gravity()
        result["force_API_probe_role"] = ROLE
        return result
    def runner(cohort, device):
        result = base.common.StrictPPOExportRunner(cohort, device)
        result.inference_provenance = {**base.provenance(cohort), "kind": KIND, "role": ROLE}
        return result
    def finish(report, contract, plan, raw):
        base._finish(report, raw, plan, contract=contract)
        arrays, diagnostics = plan["observer"].finish(raw)
        target = plan["output"]/"probe_arrays.npz"
        require(not target.exists(), "Never overwrite probe arrays")
        np.savez_compressed(target, **arrays)
        for audit in (report["ground_contact_v2_evaluation"], contract["ground_contact_v2_evaluation"]):
            audit.update(kind=KIND, role=ROLE, original_A_formal_result=False)
        report["ground_force_API_probe"] = {"kind": KIND, "role": ROLE, "arrays_sha256": sha256(target),
            "wrapper_sha256": sha256(__file__), "measurement_only": True, "diagnostics": diagnostics}
        write_json(plan["input_dir"]/"evaluation_source_audit.json", report["ground_contact_v2_evaluation"])
        proof = {"kind": KIND, "role": ROLE, "status": "completed_measurements_not_A_formal_result",
            "mu": .7, "samples": SAMPLES, "native_states": 2793, "duration_s": 6.98,
            "actor_sha256": ACTOR_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
            "source_sha256": {str(Path(__file__).resolve()): sha256(__file__), str(Path(base.__file__).resolve()): BASE_SHA,
                str(Path(collector.__file__).resolve()): COLLECTOR_SHA, str(Path(physx.__file__).resolve()): sha256(physx.__file__)},
            "generated_source_sha256": plan["generated_sha256"], "probe_arrays_sha256": sha256(target),
            "physics_trajectory_sha256": sha256(plan["output"]/"physics_trajectory.npz"),
            "report_sha256_expected_on_disk": hashlib.sha256((json.dumps(report, indent=2, allow_nan=False)+"\n").encode()).hexdigest(),
            "diagnostics": diagnostics, "USD_gravity_optional_readback": plan["usd_gravity"],
            "no_reward_owner_installed": True, "env_step_not_wrapped_by_probe": True,
            "no_actor_observation_action_state_or_physics_override": True,
            "additional_sensor_configuration_only": {"legacy_per_wheel_capacity": 32, "new_per_wheel_capacity": 64},
            "height_or_flight_failure_label": False, "original_A_strict_loader_must_reject": True,
            "limitations": ["Standalone diagnostic episode, not a replacement for A's formal comparison.",
                "Capacity flags and force-API differences are measurements, not assumptions about physical flight.",
                "No reward return or controller superiority comparison; force values remain unnormalized N."]}
        write_json(plan["output"]/"probe_audit.json", proof)
    namespace = {"__name__": "_A_actor_force_API_probe", "__file__": str(Path(__file__).resolve()),
        "_new_plan": plan, "_new_configure": configure, "_new_audit_materials": audit_materials,
        "_new_runner": runner, "_new_finish_report": finish}
    previous = sys.argv
    try:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(generated, "<A_actor_force_API_probe>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = previous


if __name__ == "__main__": run(sys.argv[1:])
