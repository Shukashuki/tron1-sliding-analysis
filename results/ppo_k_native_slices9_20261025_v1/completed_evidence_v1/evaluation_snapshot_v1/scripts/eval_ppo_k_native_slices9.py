"""Evaluate the separately trained native-slices9 actor without PPO updates.

Training geometry and validation geometry are recorded separately. Both
validation arms use explicit 1 mm contact and zero rest offsets, unchanged
authored mass properties, native Coulomb forces and the full reference.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import verify_ppo_k_native_slices9 as verification
import ppo_native_slices9_model as training_model
import tron1_wheel_geometry as geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_parent(k_snapshot):
    namespace = verification.parent_evaluator(k_snapshot)
    # ROOT stays bound to the immutable dependency snapshot; the executing
    # wrapper's identity in prepare_plan/validate_output is this file.
    namespace["__file__"] = str(Path(__file__).resolve())
    namespace["KIND"] = "model_specific_native_slices9_ppo_evaluation"
    namespace["verify_training"] = lambda folder, protocol, allow_smoke=False: verification.verify_training(
        folder, protocol, k_snapshot, allow_smoke=allow_smoke)
    return SimpleNamespace(**namespace)


def validate_output(folder, training_dir, protocol, k_snapshot, *, validation_geometry, allow_smoke=False):
    folder = Path(folder)
    parent = load_parent(k_snapshot)
    proof = parent.validate_output(folder, training_dir, protocol, backend="physx", allow_smoke=allow_smoke)
    cohort = verification.verify_training(training_dir, protocol, k_snapshot, allow_smoke=allow_smoke)
    metadata = json.loads((folder/"model_evaluation_status.json").read_text())
    report = json.loads((folder/"report.json").read_text())
    assert report["model_specific_slices9_evaluation"] == metadata
    assert report["policy_export"]["validation_wheel_contact_model"] == metadata
    assert metadata["kind"] == "model_specific_native_slices9_ppo_evaluation"
    assert metadata["status"] == "completed_requires_analysis"
    assert metadata["training_model_id"] == training_model.MODEL_ID
    assert metadata["training_cohort_id"] == training_model.COHORT_ID
    assert metadata["training_model"] == cohort["protocol"]["wheel_contact_model"]
    assert metadata["model_training_audit_sha256"] == cohort["model_training_audit_sha256"]
    assert metadata["normalized_actor_sha256"] == cohort["input_sha256"]["policy.pt"]
    assert validation_geometry in ("authored", "slices9") and metadata["validation_geometry"] == validation_geometry
    assert metadata["validation_force_law"] == "native_coulomb" and metadata["mu"] == .7
    assert metadata["validation_contact_offset_m"] == .001 and metadata["validation_rest_offset_m"] == 0.
    assert metadata["no_retraining_or_reward_action_reference_change_during_validation"]
    assert metadata["free_absolute_height"] and metadata["single_wheel_support_allowed"]
    assert metadata["all_physical_samples_retained"] and not metadata["task_success_claimed"]
    assert all(sha(path) == digest for path, digest in metadata["source_sha256"].items())
    for name in ("physics_trajectory", "tire_trajectory"):
        assert sha(folder/(name+".npz")) == metadata[name+"_sha256"]
    native = metadata["native_audit"]
    expected = 9 if validation_geometry == "slices9" else 1
    assert native["masses_COM_inertias_armature_damping_match_authored"]
    assert native["shapes_per_wheel"] == expected and native["total_native_shapes"] == 7+2*expected
    for wheel in geometry.WHEELS:
        slots = native["shape_mapping"][wheel]
        assert len(slots) == expected
        np.testing.assert_allclose(np.asarray(native["contact_offsets_m"])[:, slots], .001, rtol=0, atol=1e-8)
        np.testing.assert_allclose(np.asarray(native["rest_offsets_m"])[:, slots], 0., rtol=0, atol=1e-8)
    with np.load(folder/"tire_trajectory.npz", allow_pickle=False) as tire:
        assert tire["time_before_step_s"].shape == (2792,)
        np.testing.assert_allclose(tire["time_before_step_s"], np.arange(2792)*.0025, rtol=0, atol=1e-12)
        assert np.max(tire["native_patch_count"]) < 128
        np.testing.assert_array_equal(tire["applied_tangent_force_w_n"], np.zeros((2792, 2, 3)))
        np.testing.assert_array_equal(tire["applied_tangent_moment_COM_w_nm"], np.zeros((2792, 2, 3)))
    return dict(**proof, validation_geometry=validation_geometry, model_id=training_model.MODEL_ID,
        model_metadata_sha256=sha(folder/"model_evaluation_status.json"), task_success_claimed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--k-snapshot", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--authored-native-report", type=Path, required=True)
    parser.add_argument("--geometry", choices=("authored", "slices9"), required=True)
    parser.add_argument("--validate-only", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    assert sha(args.authored_native_report) == training_model.BASELINE_REPORT_SHA
    parent = load_parent(args.k_snapshot)
    if args.validate_only:
        inherited = parent.parser().parse_args(remaining)
        assert inherited.backend == "physx"
        proof = validate_output(inherited.output_dir, inherited.training_dir, inherited.protocol,
            args.k_snapshot, validation_geometry=args.geometry, allow_smoke=inherited.allow_smoke_policy)
        print(json.dumps(proof))
        return proof
    plan = parent.prepare_plan(remaining)
    assert plan["args"].backend == "physx" and plan["mu"] == .7
    assert plan["cohort"]["model_training_audit"]["model_id"] == training_model.MODEL_ID
    sys.path.append(str(args.runtime_source.resolve()))
    import physx_tire_runtime as tire
    import eval_ppo_backend_400hz_isaac as physx
    assert sha(tire.__file__) == "9b7dd75cbb937496807d74ebda24318aa7016a975c0723233bbec4797b8889fa"
    baseline = json.loads(args.authored_native_report.read_text())["contact_material_audit"]["native_plant_readback"]
    plan["wrapper_sha256"] = sha(__file__)
    generated = physx.materialize_source(plan["evaluator"].read_text(), plan["legacy_root"])
    plan["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
    geometry_audits, native_audits, runtimes = [], [], []
    sources = {str(Path(path).resolve()): sha(path) for path in (
        __file__, verification.__file__, training_model.__file__, geometry.__file__,
        tire.__file__, Path(tire.__file__).with_name("nonlinear_tire_contact.py"), args.authored_native_report)}

    def configure(cfg, profile, current):
        physx.configure(cfg, profile, current)
        tire.configure_sensors(cfg.scene)
        assert cfg.scene.num_envs == 1
        original = cfg.scene.robot.spawn.func

        def spawn(path, spawn_cfg, *a, **kw):
            prim = original(path, spawn_cfg, *a, **kw)
            assert not geometry_audits, "Validation is strictly single environment"
            geometry_audits.extend(geometry.apply_wheel_geometry(
                prim.GetStage(), str(prim.GetPath()), level=args.geometry,
                contact_offset_m=.001, rest_offset_m=0.))
            return prim
        cfg.scene.robot.spawn.func = spawn

    def audit_materials(env, current):
        import torch
        from physx_mjlab_recipe import _shape_map
        torch.cuda.set_device(current["args"].physical_gpu_index)
        observed = physx.audit_materials(env, current)
        plant = observed["native_plant_readback"]
        assert plant["body_names"] == baseline["body_names"] and plant["joint_names"] == baseline["joint_names"]
        for name in ("get_masses", "get_inertias", "get_coms", "get_dof_armatures", "get_dof_dampings"):
            np.testing.assert_allclose(plant["readbacks"][name]["values"], baseline["readbacks"][name]["values"], rtol=0, atol=1e-7)
        robot = env.scene["robot"]
        view = robot.root_physx_view
        mapping = _shape_map(robot)
        expected = 9 if args.geometry == "slices9" else 1
        assert view.count == 1 and view.max_shapes == 7 + expected*2
        contact, rest = tire.numpy(view.get_contact_offsets()), tire.numpy(view.get_rest_offsets())
        assert contact.shape == rest.shape == (1, view.max_shapes)
        for wheel in geometry.WHEELS:
            slots = mapping[wheel]
            assert len(slots) == expected
            np.testing.assert_allclose(contact[:, slots], .001, rtol=0, atol=1e-8)
            np.testing.assert_allclose(rest[:, slots], 0., rtol=0, atol=1e-8)
        assert not native_audits and not runtimes
        native_audits.append(dict(shape_mapping=mapping, shapes_per_wheel=expected,
            total_native_shapes=view.max_shapes, contact_offsets_m=contact.tolist(), rest_offsets_m=rest.tolist(),
            masses_COM_inertias_armature_damping_match_authored=True))
        runtimes.append(tire.TireRuntime(env, law="native_coulomb", mu=.7))
        return observed

    def runner(cohort, device):
        result = parent.common.StrictPPOExportRunner(cohort, device)
        result.inference_provenance = parent.provenance(cohort)
        return result

    def finish(report, contract, current, raw):
        parent._finish(report, raw, current, contract=contract)
        output = current["output"]
        runtime = runtimes[0].save(output/"tire_trajectory.npz")
        with np.load(output/"tire_trajectory.npz", allow_pickle=False) as data:
            assert np.max(data["native_patch_count"]) < 128
        assert all(sha(path) == digest for path, digest in sources.items()), "Evaluation source changed"
        metadata = dict(kind="model_specific_native_slices9_ppo_evaluation", status="completed_requires_analysis",
            training_cohort_id=training_model.COHORT_ID, training_model_id=training_model.MODEL_ID,
            training_model=copy.deepcopy(current["cohort"]["protocol"]["wheel_contact_model"]),
            validation_geometry=args.geometry, validation_force_law="native_coulomb", mu=.7,
            validation_contact_offset_m=.001, validation_rest_offset_m=0.,
            normalized_actor_sha256=current["cohort"]["input_sha256"]["policy.pt"],
            model_training_audit_sha256=current["cohort"]["model_training_audit_sha256"],
            native_audit=native_audits[0], geometry_audits=geometry_audits, runtime=runtime,
            physics_dt_s=.0025, policy_dt_s=.02, native_states=2793, policy_decisions=349,
            physics_trajectory_sha256=sha(output/"physics_trajectory.npz"),
            tire_trajectory_sha256=sha(output/"tire_trajectory.npz"), source_sha256=sources,
            no_retraining_or_reward_action_reference_change_during_validation=True,
            single_wheel_support_allowed=True, free_absolute_height=True,
            all_physical_samples_retained=True, task_success_claimed=False)
        report["model_specific_slices9_evaluation"] = metadata
        contract["validation_wheel_contact_model"] = copy.deepcopy(metadata)
        with (output/"model_evaluation_status.json").open("x") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)

    namespace = dict(__name__="_model_specific_slices9_physx_evaluation", __file__=str(Path(__file__).resolve()),
        _new_plan=plan, _new_configure=configure, _new_audit_materials=audit_materials,
        _new_runner=runner, _new_finish_report=finish)
    previous = sys.argv
    try:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(generated, "<model_specific_slices9_physx_evaluation>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main(sys.argv[1:])
