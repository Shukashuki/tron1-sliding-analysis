"""Observe native contact for frozen K/N PPO; require byte-identical baseline.

Fixed400 Hz/μ=.7 native cylinder friction. Additional contact-position/patch
readback only; no tire-law, material, reference, reward, action or state change.
This is a diagnostic replay, not another training run or an independent seed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


EVALUATOR_PINS = {
    "v2k": "cbb17fc4d2a7b1a4033fc8011222c8f2d99616ace4cba504c9e2537f53f5020c",
    "v2n": "0bfe76419af65bcc9caf14fff8e7d5484006eb34aea109722c4d7cc819b325c8",
}
RUNTIME_PIN = "9b7dd75cbb937496807d74ebda24318aa7016a975c0723233bbec4797b8889fa"
ARITHMETIC_PIN = "66502784ac1ff000295a24987a3fca3f5f3acb4a38192ae2a6f4fd5ea4066410"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--base-snapshot", type=Path, required=True)
    result.add_argument("--cohort", choices=tuple(EVALUATOR_PINS), required=True)
    result.add_argument("--baseline-dir", type=Path, required=True)
    return result


def main():
    args, remaining = parser().parse_known_args()
    scripts = args.base_snapshot.resolve()/"scripts"
    name = "eval_ppo_ground_contact_"+args.cohort
    evaluator_path = scripts/(name+".py")
    assert sha256(evaluator_path) == EVALUATOR_PINS[args.cohort], "Frozen parent evaluator differs"
    sys.path.append(str(scripts))
    base = importlib.import_module(name)
    assert Path(base.__file__).resolve() == evaluator_path
    import eval_ppo_backend_400hz_isaac as physx
    import physx_tire_runtime as tire
    assert sha256(tire.__file__) == RUNTIME_PIN
    assert sha256(Path(tire.__file__).with_name("nonlinear_tire_contact.py")) == ARITHMETIC_PIN
    plan = base.prepare_plan(remaining)
    assert plan["args"].backend == "physx" and plan["mu"] == .7
    baseline = args.baseline_dir.resolve()
    proof = base.validate_output(baseline, plan["args"].training_dir,
        plan["args"].protocol, backend="physx", allow_smoke=False)
    baseline_recording = baseline/"physics_trajectory.npz"
    baseline_hash = sha256(baseline_recording)
    assert baseline_hash == proof["recording_sha256"]
    generated = physx.materialize_source(plan["evaluator"].read_text(), plan["legacy_root"])
    plan["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
    runtime = []

    def configure(cfg, profile, current):
        physx.configure(cfg, profile, current)
        tire.configure_sensors(cfg.scene)

    def audit_materials(env, current):
        import torch
        torch.cuda.set_device(current["args"].physical_gpu_index)
        original = physx.audit_materials(env, current)
        assert not runtime and env.physics_dt == .0025
        runtime.append(tire.TireRuntime(env, law="native_coulomb", mu=.7))
        assert not runtime[0].material_audit["native_wheel_tangential_friction_disabled"]
        return {"nominal_unchanged_materials": original,
                "additional_native_contact_observer": runtime[0].material_audit}

    def runner(cohort, device):
        result = base.common.StrictPPOExportRunner(cohort, device)
        result.inference_provenance = base.provenance(cohort)
        return result

    def finish(report, contract, current, raw):
        output = current["output"]
        base.common.verify_recording(output/"physics_trajectory.npz", "PhysX")
        assert report["status"] == "evaluated" and report["deterministic_start_frame"] == 0
        assert report["runtime_device_audit"]["status"] == "verified"
        observed = sha256(output/"physics_trajectory.npz")
        assert observed == baseline_hash, "Observer changed native recording; do not interpret its contact data"
        assert sha256(baseline_recording) == baseline_hash, "Original baseline changed during observation"
        observation = runtime[0].save(output/"native_contact_detail.npz")
        metadata = {
            "kind": "native_PhysX_contact_observation_identical_closed_loop_replay",
            "status": "completed_requires_contact_dynamics_analysis", "cohort": args.cohort,
            "normalized_actor_sha256": current["cohort"]["input_sha256"]["policy.pt"],
            "native_states": 2793, "policy_decisions": 349, "physics_dt_s": .0025, "mu": .7,
            "baseline_recording_sha256": baseline_hash, "baseline_proof": proof,
            "baseline_recording_byte_identical": True, "baseline_directory": str(baseline),
            "native_contact_detail_sha256": sha256(output/"native_contact_detail.npz"),
            "physics_trajectory_sha256": observed, "observer": observation,
            "no_policy_training_tire_or_reward_change": True, "one_environment": True,
            "observations_are_not_an_independent_seed": True,
            "friction_patch_sum_is_not_pointwise_cone_identification": True,
            "source_sha256": {str(path.resolve()): sha256(path) for path in
                (Path(__file__), Path(base.__file__), Path(tire.__file__),
                 Path(tire.__file__).with_name("nonlinear_tire_contact.py"))},
            "task_success_claimed": False,
        }
        report["native_contact_diagnostic"] = metadata
        contract["native_contact_diagnostic"] = metadata
        with (output/"native_contact_diagnostic.json").open("x") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)

    namespace = {"__name__": "_native_contact_diagnostic", "__file__": str(Path(__file__).resolve()),
        "_new_plan": plan, "_new_configure": configure, "_new_audit_materials": audit_materials,
        "_new_runner": runner, "_new_finish_report": finish}
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(generated, "<native_contact_diagnostic>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = previous


if __name__ == "__main__": main()
