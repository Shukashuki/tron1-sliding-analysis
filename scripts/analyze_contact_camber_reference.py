"""Matched3D-axis reward with old versus independently tilted wheel references."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest()=="cd11355c2cfa6834706a75c8f460c327f363ac805bf9c4db55393221e4fab8dc"
    for old,new in [
        ("import analyze_contact_taskspace_refinement as control","import analyze_contact_wheel_axis as control"),
        ("import car_taskspace_tracking as old_targets","import car_wheel_axis_tracking as old_targets"),
        ("import collect_contact_wheel_axis as collection","import collect_contact_camber_reference as collection"),
        ('"axis_physx", "control_mujoco", "axis_mujoco"','"camber_physx", "control_mujoco", "camber_mujoco"'),
        ('results/contact_taskspace_refinement_20261025_v2/completed_evidence_v1','results/contact_wheel_axis_20261025_v1/completed_evidence_v1'),
        ('source_snapshot_v1/config/contact_wheel_axis_400hz_v1.json','source_snapshot_v4/config/contact_camber_reference_400hz_v1.json'),
        ('for key in ("asset", "model", "motion"):', 'for key in ("asset", "model"): '),
        ('target_hash(before, "car_taskspace_tracking.py")','target_hash(before, "car_wheel_axis_tracking.py")'),
        ('dict(control=old_proofs["taskspace"], axis=driver["proofs"]["formal_K"])',
         'dict(control=old_proofs["axis"], camber=driver["proofs"]["formal_K"])'),
        ('proofs["axis"][','proofs["camber"]['),
        ('r["id"] == "taskspace_"+engine','r["id"] == "axis_"+engine'),
        ('old_traces["taskspace_"+engine]','old_traces["axis_"+engine]'),
        ('"2D wheel heading | "','"Original reference /3D axle | "'),
        ('"wheel_axis_refinement_native_evaluation"','"camber_reference_native_evaluation"'),
        ('("position", "com_velocity", "angular_velocity", "joint_pos", "joint_vel")',
         '("position", "com_velocity", "angular_velocity", "joint_pos")'),
        ('dict(id="axis_"+engine, variant="axis", engine=engine, title="3D wheel axis | "+engine,',
         'dict(id="camber_"+engine, variant="camber", engine=engine, title="Camber reference /3D axle | "+engine,\n'
         '                   initial_joint_velocity_max_delta_vs_control_rad_s=float(abs(trace["joint_vel"][0]-other["joint_vel"][0]).max()),'),
        ('"matched_2D_heading_3D_axis_refinement_dual_engine"','"matched_original_independent_camber_reference_dual_engine"'),
        ('only_changed_reward_definition="motion_body_ori:2D_axial_heading_to_3D_axial_direction",',
         'only_changed_input="motion_NPZ:independent_camber_reconstructed_kinematics",'),
        ('contact_formula_all_weights_and_other14_rewards_unchanged=True,',
         'all15_reward_definitions_weights_contact_DR_actuators_unchanged=True,'),
        ('"2D heading /3D axle | matched K warm refinement | mu=.7"',
         '"Original /independent camber reference | same K checkpoint,3D reward,budget | mu=.7"'),
        ('    proofs = dict(control=',
         '    assert before["inputs"]["motion"]["sha256"]=="da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"\n'
         '    assert after["inputs"]["motion"]["sha256"]=="0c8f2cfc2876bbd103bad206e708d91bf6753f8faa75c3107b0ec37c56c448b9"\n'
         '    proofs = dict(control='),
        ('    rows.sort(key=lambda r: ORDER.index(r["id"]))',
         '    for name in ("position","com_velocity","angular_velocity","joint_pos","joint_vel"):\n'
         '        np.testing.assert_allclose(traces["camber_physx"][name][0],traces["camber_mujoco"][name][0],atol=2e-6,rtol=0)\n'
         '    rows.sort(key=lambda r: ORDER.index(r["id"]))'),
    ]:
        assert source.count(old)>=1,old;source=source.replace(old,new)
    compile(source,"<camber_reference_numerical_comparison>","exec");return source


_source=materialize((Path(__file__).parent/"analyze_contact_wheel_axis.py").read_text())
_namespace=dict(__name__="_camber_reference_numerical_comparison",__file__=str(Path(__file__).resolve()))
exec(compile(_source,"<camber_reference_numerical_comparison>","exec"),_namespace)
for _name in ("ROOT","ORDER","CONTROL","metrics","read","sha","npz","load_cases","run"):
    globals()[_name]=_namespace[_name]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("collection","reference","output-dir"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
