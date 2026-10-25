"""Recompute S/T native tracking and unchanged nine-gate contact acceptance."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()=="b62e61b551e01acd3ad1a46907025ddd0706da6c7a069bfd6c95a51832ca9e7b"
    changes=[
        ("import collect_contact_warm_refinement as collection","import collect_contact_duration_pair as collection"),
        ("collection.INITIAL","collection.namespace()['INITIAL']"),
        ("ORDER=('K_physx','M_physx','K_mujoco','M_mujoco')","ORDER=('S_physx','T_physx','S_mujoco','T_mujoco')"),
        ("source_snapshot_v1/config/contact_warm_refinement_400hz_v1.json","source_snapshot_v3/config/contact_duration_pair_400hz_v1.json"),
        ("['K','M']","['S','T']"),
        ("manifests['K'],manifests['M']","manifests['S'],manifests['T']"),
        ("formal_K","formal_S"),("formal_M","formal_T"),
        ("['position','com_velocity','angular_velocity','joint_pos','joint_vel']","['position','com_velocity','angular_velocity','joint_pos']"),
        ("paired_warm_K_M_contact_refinement_dual_engine","matched_support_duration_S_T_dual_engine"),
        ("Paired warm K/M | same K checkpoint + same conservative2.46M refinement | mu=.7","Matched S/T | worst+mean versus twice worst | same K checkpoint +2.46M each | mu=.7"),
        ("    return driver,status,protocol",
         "    from camber_reference_contract import MOTION,DECLARATION\n"
         "    assert protocol['kinematic_reference']==DECLARATION\n"
         "    assert protocol['variant_contact_settings']['S']['support_gap_aggregation']=='max8_plus_mean8'\n"
         "    assert protocol['variant_contact_settings']['T']['support_gap_aggregation']=='max8'\n"
         "    assert protocol['variant_contact_settings']['S']['support_gap_weight']==1.\n"
         "    assert protocol['variant_contact_settings']['T']['support_gap_weight']==2.\n"
         "    for variant,manifest in manifests.items():\n"
         "        proof=driver['proofs']['formal_'+variant]\n"
         "        assert proof['matched_scale_duration_verified'] and proof['independent_camber_reference_verified']\n"
         "        assert manifest['inputs']['motion']['sha256']==MOTION\n"
         "        for stage in ['live_startup','live_final']:\n"
         "            a=manifest['cohort'][stage]['contact']\n"
         "            assert a['matched_scale_duration_variant']==variant and a['raw_contact_reward_bounds']==[-2.,1.]\n"
         "    return driver,status,protocol"),
        ("    rows.sort(key=lambda r:ORDER.index(r['id']))",
         "    for name in ['position','com_velocity','angular_velocity','joint_pos','joint_vel']:\n"
         "        for key in ORDER[1:]:np.testing.assert_allclose(traces[key][name][0],traces[ORDER[0]][name][0],atol=2e-6,rtol=0)\n"
         "    for key in ORDER[1:]:assert abs(np.dot(traces[key]['quaternion'][0],traces[ORDER[0]]['quaternion'][0]))>=1-1e-7\n"
         "    rows.sort(key=lambda r:ORDER.index(r['id']))"),
        ("        same_K_initialization_and_optimizer_between_arms=True,all_original_gates_retained=True,",
         "        same_K_initialization_and_optimizer_between_arms=True,all_original_gates_retained=True,\n"
         "        matched_raw_cost_bounds=[0.,2.],same_positive_G_and_other14_terms=True,\n"
         "        scoring_reference='original_locked_car_body_path',native_motion_reference='independent_camber_0c8f2cfc',\n"
         "        same_independent_camber_reference=True,only_reward_intervention='worst_plus_mean_vs_twice_worst',"),
    ]
    for old,new in changes:
        assert old in source,old
        source=source.replace(old,new)
    compile(source,"<matched_duration_pair_analysis>","exec");return source


_ns=dict(__name__="_matched_duration_pair_analysis",__file__=str(Path(__file__).resolve()))
exec(compile(materialize((Path(__file__).parent/"analyze_contact_warm_refinement.py").read_text()),"<matched_duration_pair_analysis>","exec"),_ns)
for _name in ("ROOT","ORDER","metrics","read","sha","npz","verify_pair","load_cases","run"):
    globals()[_name]=_ns[_name]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("collection","reference","output-dir"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
