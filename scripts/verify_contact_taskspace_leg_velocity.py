"""Strict original warm-start admission plus R physical-velocity audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import car_taskspace_tracking as targets
import train_contact_taskspace_leg_velocity as training

PARENT_SHA="8db079293c5f752166d01fb71c0170c32e4db0e6649c151e6f8ebf1c25e226f1"


def verify_training(directory,protocol_path,*,allow_smoke=False,cpu_inference=False):
    p=Path(__file__).parent/"verify_contact_warm_refinement.py"
    assert training.sha(p)==PARENT_SHA
    source=p.read_text()
    pairs=[("import train_contact_warm_refinement as training","import train_contact_taskspace_leg_velocity as training"),
           ("variant in ['K','M']","variant in ['R']")]
    for old,new in pairs:
        assert source.count(old)==1;source=source.replace(old,new,1)
    ns=dict(__name__="_R_original_strict_warm_admission",__file__=str(p))
    exec(compile(source,"<R_original_strict_warm_admission>","exec"),ns)
    c=ns["verify_training"](directory,protocol_path,allow_smoke=allow_smoke,cpu_inference=cpu_inference)
    assert c["protocol"]["variants"]==["R"] and c["variant"]=="R"
    assert c["contract"]["task_profile"]["task_space_tracking"]==targets.settings()
    for key in ("live_startup","live_final"):
        live=c["manifest"]["cohort"][key]
        training.velocity_audit(live["contact"])
        a=live["task_space_tracking"]
        assert a["status"]=="live_target_selection_verified" and a["configured"]["settings"]==targets.settings()
        assert a["configured"]["four_tracking_terms_changed"]==list(targets.CHANGED)
        assert a["active_tracking_params"]["motion_body_pos"]["body_names"]==targets.SETTINGS["position_bodies"]
        assert a["active_tracking_params"]["motion_body_lin_vel"]["body_names"]==targets.SETTINGS["linear_velocity_bodies"]
        assert a["active_tracking_params"]["motion_body_ang_vel"]["body_names"]==["base_Link"]
        assert a["active_tracking_params"]["motion_body_ori"]["body_names"]==targets.SETTINGS["wheel_bodies"]
        assert a["active_tracking_params"]["motion_body_ori"]["std"]==.4
        assert a["extra_simulation_steps"]==0 and not a["observation_layout_changed"] and not a["absolute_height_gate"]
    c["actual_leg_velocity_audit_verified"]=True
    return c


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--training-dir",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--allow-smoke",action="store_true");p.add_argument("--cpu-inference",action="store_true")
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    c=verify_training(a.training_dir,a.protocol,allow_smoke=a.allow_smoke,cpu_inference=a.cpu_inference)
    d=c["completion"]
    result=dict(kind="strict_R_actual_leg_velocity_warm_verification",status="completed_verified",cohort_id=training.COHORT_ID,
        variant="R",run_kind=d["run_kind"],transitions=d["training_environment_steps"],cumulative_transitions=d["cumulative_training_environment_steps"],
        actor_sha256=c["input_sha256"]["policy.pt"],checkpoint_sha256=c["input_sha256"]["model_final.pt"],
        completion_audit_sha256=c["input_sha256"]["cohort_completion_audit.json"],initial_policy_state_sha256=training.WARM_STATE_SHA,
        CPU_actor_output_shape=c.get("CPU_actor_output_shape"),task_space_tracking_verified=True,actual_leg_velocity_audit_verified=True,
        from_scratch=False,historical_checkpoint_used=True,verifier_sha256=training.sha(__file__),task_success_claimed=False)
    with a.output.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=="__main__":main()
