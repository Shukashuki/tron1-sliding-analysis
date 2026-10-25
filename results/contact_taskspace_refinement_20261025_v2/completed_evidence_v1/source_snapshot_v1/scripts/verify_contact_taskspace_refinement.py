"""Retain all warm-start proofs and additionally verify live task-space targets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import car_taskspace_tracking as targets
import train_contact_taskspace_refinement as training

PARENT_SHA = "8db079293c5f752166d01fb71c0170c32e4db0e6649c151e6f8ebf1c25e226f1"


def verify_training(directory, protocol_path, *, allow_smoke=False, cpu_inference=False):
    source_path = Path(__file__).parent/"verify_contact_warm_refinement.py"
    assert training.sha(source_path) == PARENT_SHA
    source = source_path.read_text()
    old = "import train_contact_warm_refinement as training"
    assert source.count(old) == 1
    source = source.replace(old, "import train_contact_taskspace_refinement as training", 1)
    namespace = dict(__name__="_taskspace_strict_warm_verifier", __file__=str(source_path))
    exec(compile(source, "<taskspace_strict_warm_verifier>", "exec"), namespace)
    result = namespace["verify_training"](directory, protocol_path, allow_smoke=allow_smoke, cpu_inference=cpu_inference)
    p = result["protocol"]
    assert p["variants"] == ["K"] and result["variant"] == "K"
    expected = targets.settings(p["task_space_tracking"])
    assert result["contract"]["task_profile"]["task_space_tracking"] == expected
    for key in ("live_startup", "live_final"):
        a = result["manifest"]["cohort"][key]["task_space_tracking"]
        assert a["status"] == "live_target_selection_verified"
        assert a["configured"]["settings"] == expected
        assert a["configured"]["four_tracking_terms_changed"] == list(targets.CHANGED)
        assert a["configured"]["contact_reward_and_other_eleven_terms_unchanged"]
        assert a["active_tracking_params"]["motion_body_pos"]["body_names"] == expected["position_bodies"]
        assert a["active_tracking_params"]["motion_body_lin_vel"]["body_names"] == expected["linear_velocity_bodies"]
        assert a["active_tracking_params"]["motion_body_ang_vel"]["body_names"] == ["base_Link"]
        assert a["active_tracking_params"]["motion_body_ori"]["body_names"] == expected["wheel_bodies"]
        assert a["active_tracking_params"]["motion_body_ori"]["std"] == .4
        assert a["extra_simulation_steps"] == 0 and not a["observation_layout_changed"] and not a["absolute_height_gate"]
        assert a["live_heading_reward_shape"] == [result["manifest"]["num_envs"]]
    result["task_space_tracking_verified"] = True
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--training-dir", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--allow-smoke", action="store_true")
    p.add_argument("--cpu-inference", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    c = verify_training(a.training_dir, a.protocol, allow_smoke=a.allow_smoke, cpu_inference=a.cpu_inference)
    d = c["completion"]
    proof = dict(kind="strict_contact_taskspace_refinement_training_verification", status="completed_verified",
                 cohort_id=training.COHORT_ID, variant="K", run_kind=d["run_kind"],
                 transitions=d["training_environment_steps"], cumulative_transitions=d["cumulative_training_environment_steps"],
                 actor_sha256=c["input_sha256"]["policy.pt"], checkpoint_sha256=c["input_sha256"]["model_final.pt"],
                 completion_audit_sha256=c["input_sha256"]["cohort_completion_audit.json"],
                 initial_policy_state_sha256=training.WARM_STATE_SHA, CPU_actor_output_shape=c.get("CPU_actor_output_shape"),
                 task_space_tracking_verified=True, from_scratch=False, historical_checkpoint_used=True,
                 verifier_sha256=training.sha(__file__), task_success_claimed=False)
    with a.output.open("x") as f:
        json.dump(proof, f, indent=2, allow_nan=False)
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
