"""Frozen-policy PhysX tire-law sensitivity; a new cohort, not old M evidence.

Reuses the completed normalized M actor and the pinned400Hz evaluator. Only
wheel tangential contact is substituted. Native contacts, free body height,
joint actuation, reference, actor observations and full-horizon resets retain
the existing evaluation definitions. No policy adaptation is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--base-snapshot",type=Path,required=True)
    p.add_argument("--law",choices=("native_coulomb","external_coulomb_regularized","combined_slip_magic_formula"),required=True)
    p.add_argument("--physics-dt-s",type=float,choices=(.0025,.00125),default=.0025)
    a,remaining=p.parse_known_args()
    assert (a.base_snapshot/"scripts/eval_ppo_ground_contact_v2m.py").is_file()
    sys.path.append(str(a.base_snapshot/"scripts"))
    import eval_ppo_ground_contact_v2m as base
    import eval_ppo_backend_400hz_isaac as physx
    import physx_tire_runtime as tire
    plan=base.prepare_plan(remaining)
    assert plan["args"].backend=="physx" and plan["mu"]==.7
    if a.physics_dt_s!=.0025:
        sys.path.insert(0,str(plan["legacy_root"]))
        from training.tron1_physics_clock import apply_physics_clock_to_contract
        plan["contract"]=apply_physics_clock_to_contract(plan["contract"],a.physics_dt_s)
    generated=physx.materialize_source(plan["evaluator"].read_text(),plan["legacy_root"])
    plan["generated_sha256"]=hashlib.sha256(generated.encode()).hexdigest()
    runtime=[]

    def configure(cfg,profile,current):
        physx.configure(cfg,profile,current)
        if a.physics_dt_s!=.0025:
            from training.tron1_physics_clock import configure_physics_clock
            configure_physics_clock(cfg,a.physics_dt_s)
            for value in vars(cfg.scene).values():
                if hasattr(value,"update_period") and hasattr(value,"filter_prim_paths_expr"):
                    value.update_period=a.physics_dt_s
        tire.configure_sensors(cfg.scene)

    def audit(env,current):
        import torch
        torch.cuda.set_device(current["args"].physical_gpu_index)
        original=physx.audit_materials(env,current)
        assert not runtime
        runtime.append(tire.TireRuntime(env,law=a.law,mu=.7))
        return {"nominal_before_tire_substitution":original,"tire_substitution":runtime[0].material_audit}

    def runner(cohort,device):
        result=base.common.StrictPPOExportRunner(cohort,device)
        result.inference_provenance=base.provenance(cohort)
        return result

    def finish(report,contract,current,raw):
        if a.physics_dt_s==.0025:
            base.common.verify_recording(current["output"]/"physics_trajectory.npz","PhysX")
        else:
            import numpy as np
            expected=349*round(.02/a.physics_dt_s)+1
            assert np.array_equal(raw["valid_mask"],np.ones((expected,1),dtype=bool))
            assert np.allclose(raw["time_s"],np.arange(expected)*a.physics_dt_s,atol=1e-12,rtol=0)
            assert float(raw["physics_dt_s"])==a.physics_dt_s
        assert report["status"]=="evaluated" and report["deterministic_start_frame"]==0
        assert report["runtime_device_audit"]["status"]=="verified"
        coupling=runtime[0].save(current["output"]/"tire_trajectory.npz")
        audit={"study":"PhysX_tire_law_frozen_M_PPO_transfer","status":"completed_requires_analysis",
            "new_tire_cohort":True,"existing_M_evidence_mutated":False,"policy_retrained_for_tire_law":False,
            "law":a.law,"native_states":349*round(.02/a.physics_dt_s)+1,"policy_decisions":349,"duration_s":6.98,
            "physics_dt_s":a.physics_dt_s,"trained_physics_dt_s":.0025,
            "evaluation_only_step_refinement":a.physics_dt_s!=.0025,
            "normalized_actor_sha256":current["cohort"]["input_sha256"]["policy.pt"],
            "native_geometry_changed":False,"source_sha256":{},"runtime":coupling}
        for name in [Path(__file__),Path(tire.__file__),Path(tire.__file__).with_name("nonlinear_tire_contact.py")]:
            audit["source_sha256"][str(name)]=hashlib.sha256(name.read_bytes()).hexdigest()
        for name in ("physics_trajectory.npz","tire_trajectory.npz"):
            audit[name+"_sha256"]=hashlib.sha256((current["output"]/name).read_bytes()).hexdigest()
        report["tire_law_study"]=audit
        contract["tire_law_study"]=audit
        (current["output"]/"tire_study_status.json").write_text(json.dumps(audit,indent=2)+chr(10))

    namespace={"__name__":"_new_tire_study_eval","__file__":str(Path(__file__).resolve()),
        "_new_plan":plan,"_new_configure":configure,"_new_audit_materials":audit,
        "_new_runner":runner,"_new_finish_report":finish}
    os.environ.setdefault("CUDA_DEVICE_ORDER","PCI_BUS_ID")
    previous=sys.argv
    try:
        sys.argv=[str(Path(__file__).resolve()),*plan["inherited_argv"]]
        exec(compile(generated,"<new_PhysX_tire_study>","exec"),namespace)
        namespace["main"]()
    finally:
        sys.argv=previous


if __name__=="__main__":
    main()
