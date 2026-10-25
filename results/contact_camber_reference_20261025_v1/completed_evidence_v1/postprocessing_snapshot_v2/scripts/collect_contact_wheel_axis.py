"""Collect the complete3D-axis refinement and two native episodes,no weights."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PARENT_SHA="5a3b65b4abf7ef35b220f3ea84937a6d46a94dd789231b7157f840bdab3ecb7d"


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA
    for old,new in [
        ("COHORT='tron1_contact_warm_refinement_400hz_v1'","COHORT='tron1_contact_wheel_axis_refinement_400hz_v1'"),
        ("CASES=('K_physx','K_mujoco','M_physx','M_mujoco')","CASES=('K_physx','K_mujoco')"),
        ("both_formal_arms_completed_verified_nominal_pending","formal_completed_verified_nominal_pending"),
        ("four_native_nominal_episodes_completed_require_analysis_and_render","two_native_nominal_episodes_completed_require_analysis_and_render"),
        ("['K','M']","['K']"),
        ("['source_snapshot_v1','evaluation_snapshot_v1','postprocessing_snapshot_v1']","['source_snapshot_v1','postprocessing_snapshot_v1']"),
        ("assert not proof['from_scratch'] and proof['historical_checkpoint_used']",
         "assert not proof['from_scratch'] and proof['historical_checkpoint_used'] and proof['three_dimensional_axial_orientation_verified']"),
        ("        assert audit['training_environment_steps']==2457600",
         "        assert audit['cohort']['live_startup']['task_space_tracking']['configured']['settings']['orientation_term']=='world_axial_3D_wheel_axis'\n"
         "        assert audit['cohort']['live_final']['task_space_tracking']['configured']['settings']['orientation_term']=='world_axial_3D_wheel_axis'\n"
         "        assert audit['training_environment_steps']==2457600"),
        ("completed_paired_warm_contact_refinement_four_native_episodes","completed_3D_wheel_axis_refinement_two_native_episodes"),
    ]:
        assert old in source,old;source=source.replace(old,new)
    compile(source,"<3D_axis_completed_collection>","exec");return source


def _namespace():
    source=materialize((Path(__file__).parent/"collect_contact_warm_refinement.py").read_text())
    ns=dict(__name__="_3D_axis_completed_collection",__file__=str(Path(__file__).resolve()))
    exec(compile(source,"<3D_axis_completed_collection>","exec"),ns);return ns


def verify_completed(root):return _namespace()["verify_completed"](root)
def collect(root,helper):return _namespace()["collect"](root,helper)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--workspace",type=Path,required=True);p.add_argument("--idle-audit-source",type=Path,required=True)
    a=p.parse_args();collect(a.workspace,a.idle_audit_source)
