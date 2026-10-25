"""Collect completed S/T neural proofs and four native episodes,no weights."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()=="5a3b65b4abf7ef35b220f3ea84937a6d46a94dd789231b7157f840bdab3ecb7d"
    for old,new in [
        ("tron1_contact_warm_refinement_400hz_v1","tron1_contact_duration_credit_pair_400hz_v1"),
        ("CASES=('K_physx','K_mujoco','M_physx','M_mujoco')","CASES=('S_physx','S_mujoco','T_physx','T_mujoco')"),
        ("['K','M']","['S','T']"),
        ("['source_snapshot_v1','evaluation_snapshot_v1','postprocessing_snapshot_v1']","['source_snapshot_v3','postprocessing_snapshot_v2']"),
        ("assert not proof['from_scratch'] and proof['historical_checkpoint_used']",
         "assert not proof['from_scratch'] and proof['historical_checkpoint_used'] and proof['matched_scale_duration_verified'] and proof['independent_camber_reference_verified']"),
        ("p.suffix in ['.py','.json']","p.suffix in ['.py','.json','.npz'] and (p.suffix!='.npz' or str(p)=='references/fit_differential.npz')"),
        ("p.name in ['physics_trajectory.npz','rollout.npz']","p.name in ['physics_trajectory.npz','rollout.npz','fit_differential.npz']"),
        ("        assert audit['training_environment_steps']==2457600",
         "        assert read(folder/'policy_contract.json')['reference_file_sha256']=='0c8f2cfc2876bbd103bad206e708d91bf6753f8faa75c3107b0ec37c56c448b9'\n"
         "        assert read(folder/'manifest.json')['kinematic_reference']==read(folder/'policy_contract.json')['kinematic_reference']\n"
         "        for key in ['live_startup','live_final']:\n"
         "            a=audit['cohort'][key]['contact']\n"
         "            assert a['matched_scale_duration_variant']==variant and a['raw_contact_reward_bounds']==[-2.,1.]\n"
         "        assert audit['training_environment_steps']==2457600"),
        ("completed_paired_warm_contact_refinement_four_native_episodes","completed_matched_scale_support_duration_pair_four_native_episodes"),
    ]:
        assert old in source,old;source=source.replace(old,new)
    compile(source,"<duration_pair_completed_collection>","exec");return source


def namespace():
    ns=dict(__name__="_duration_pair_completed_collection",__file__=str(Path(__file__).resolve()))
    exec(compile(materialize((Path(__file__).parent/"collect_contact_warm_refinement.py").read_text()),"<duration_pair_completed_collection>","exec"),ns)
    return ns


def verify_completed(root):return namespace()["verify_completed"](root)
def collect(root,helper):return namespace()["collect"](root,helper)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--workspace",type=Path,required=True);p.add_argument("--idle-audit-source",type=Path,required=True)
    a=p.parse_args();collect(a.workspace,a.idle_audit_source)
