"""Matched R refinement: M plus actual six-leg velocity regularization."""
from __future__ import annotations

from pathlib import Path
import sys

import train_contact_taskspace_margin as parent
import ground_contact_reward_v2r as definition

COHORT_ID = "tron1_contact_taskspace_leg_velocity_400hz_v1"
PARENT_SHA = "c7c74aead111a3dc7a38604c46a4677689aeba2cac49a5099db5d680a417951b"
for _name in ("CHECKPOINT_SHA","ACTOR_SHA","WARM_STATE_SHA","RANDOM_STATE_SHA","BASE_TRANSITIONS",
              "RUNNER_SHA","PPO_SHA","REFINEMENT","sha","read","write","state_sha","refinement_config"):
    globals()[_name] = getattr(parent,_name)


def velocity_audit(a):
    assert a["settings"]==definition.validate_settings() and a["reward_definition_version"]=="ground_contact_v2r_any_wheel"
    assert a["device_finite_leg_velocity_health"] and a["added_actual_leg_velocity_cost"]
    assert a["leg_velocity_physics_samples"]==a["physics_samples"]==8*a["steps_completed"]
    assert a["leg_velocity_unique_reads"]==a["unique_reward_windows"]
    assert a["sampled_leg_environment_substeps"]==a["sampled_environment_substeps"]
    assert a["leg_joint_names"]==definition.LEGS and not a["wheel_spin_penalized"]
    assert a["leg_velocity_cost_ungated"] and a["extra_environment_steps"]==0


def materialize(source):
    assert sha(parent.__file__)==PARENT_SHA
    source = parent.materialize(source)
    for old,new in [
        ("COHORT_ID='"+parent.COHORT_ID+"'","COHORT_ID='"+COHORT_ID+"'"),
        ("assert protocol['variants']==['M']","assert protocol['variants']==['R']"),
        ("for variant in ['M']:","for variant in ['R']:"),
        ("choices=['M']","choices=['R']"),
        ("M margin4 contact reward; free intermediate leg-link targets","R M-plus-actual-leg-velocity reward; free intermediate leg-link targets"),
        ("            owner=current['contact_owner'];a=owner.audit(health=True)\n",
         "            owner=current['contact_owner'];a=owner.audit(health=True)\n            _R_velocity_audit(a)\n"),
    ]:
        assert source.count(old)==1,old
        source=source.replace(old,new,1)
    compile(source,"<taskspace_R_actual_leg_velocity_parent>","exec")
    return source


def source_ledger():
    here=Path(__file__).resolve().parent
    files=[Path(__file__),Path(parent.__file__),Path(parent.parent.__file__),Path(parent.parent.targets.__file__),
           Path(definition.__file__),here/"ppo_ground_contact_v2r.py",here/"verify_contact_taskspace_leg_velocity.py"]
    # These actual imported copies travel with the new snapshot. Bind them,
    # not just the equal-byte originals in the remote frozen K workspace.
    files += [here/n for n in ("ground_contact_reward_v2.py","ground_contact_reward_v2b.py",
        "ground_contact_reward_v2c.py","ground_contact_reward_v2d.py","ground_contact_reward_v2f.py",
        "ground_contact_reward_v2g.py","ground_contact_reward_v2i.py","ground_contact_reward_v2k.py",
        "ground_contact_reward_v2m.py","ppo_ground_contact_v2m.py","ground_contact_beta_contract.py")]
    return {str(p.resolve()):sha(p) for p in files}


def namespace():
    frozen=parent.parent.frozen
    ns=dict(__name__="_taskspace_R_actual_leg_velocity_parent",__file__=frozen.__file__,
            _taskspace_validate=parent.parent.validate_targets,_taskspace_sources=source_ledger,
            _taskspace_configure=parent.parent.targets.configure,_taskspace_audit=parent.parent.targets.runtime_audit,
            _R_velocity_audit=velocity_audit)
    exec(compile(materialize(Path(frozen.__file__).read_text()),"<taskspace_R_actual_leg_velocity_parent>","exec"),ns)
    return ns


if __name__=="__main__":
    namespace()["run"](sys.argv[1:])
