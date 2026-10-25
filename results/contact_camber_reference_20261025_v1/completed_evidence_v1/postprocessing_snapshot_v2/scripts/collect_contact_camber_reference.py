"""Collect completed camber-reference training and native records,no weights."""
from __future__ import annotations
import argparse
from pathlib import Path

import collect_contact_wheel_axis as parent


def materialize(source):
    source=parent.materialize(source)
    for old,new in [
        ("tron1_contact_wheel_axis_refinement_400hz_v1","tron1_contact_camber_reference_400hz_v1"),
        ("completed_3D_wheel_axis_refinement_two_native_episodes","completed_independent_camber_reference_two_native_episodes"),
        ("['source_snapshot_v1','postprocessing_snapshot_v1']","['source_snapshot_v4','postprocessing_snapshot_v2']"),
        ("proof['three_dimensional_axial_orientation_verified']", "proof['three_dimensional_axial_orientation_verified'] and proof['independent_camber_reference_verified']"),
        ("p.suffix in ['.py','.json']","p.suffix in ['.py','.json','.npz'] and (p.suffix!='.npz' or str(p)=='references/fit_differential.npz')"),
        ("p.name in ['physics_trajectory.npz','rollout.npz']","p.name in ['physics_trajectory.npz','rollout.npz','fit_differential.npz']"),
        ("        assert audit['training_environment_steps']==2457600",
         "        assert read(folder/'policy_contract.json')['reference_file_sha256']=='0c8f2cfc2876bbd103bad206e708d91bf6753f8faa75c3107b0ec37c56c448b9'\n"
         "        assert read(folder/'manifest.json')['kinematic_reference']==read(folder/'policy_contract.json')['kinematic_reference']\n"
         "        for key in ['live_startup','live_final']:\n"
         "            assert audit['cohort'][key]['kinematic_reference']['status']=='native_independent_camber_reference_tensors_verified'\n"
         "        assert audit['training_environment_steps']==2457600"),
    ]:
        assert source.count(old)>=1,old;source=source.replace(old,new)
    compile(source,"<camber_reference_completed_collection>","exec");return source


def _namespace():
    ns=dict(__name__="_camber_reference_completed_collection",__file__=str(Path(__file__).resolve()))
    exec(compile(materialize((Path(__file__).parent/"collect_contact_warm_refinement.py").read_text()),"<camber_reference_completed_collection>","exec"),ns)
    return ns


def verify_completed(root):return _namespace()["verify_completed"](root)
def collect(root,helper):return _namespace()["collect"](root,helper)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--workspace",type=Path,required=True);p.add_argument("--idle-audit-source",type=Path,required=True)
    a=p.parse_args();collect(a.workspace,a.idle_audit_source)
