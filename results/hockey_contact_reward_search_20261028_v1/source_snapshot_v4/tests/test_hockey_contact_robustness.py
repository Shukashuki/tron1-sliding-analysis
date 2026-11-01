"""Counterexamples that would invalidate a claimed 33-case success rate."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_hockey_contact_robustness import check_case_list,check_reset
from hockey_speed_study import evaluation_cases,perturbed_root


class ContactRobustnessTests(unittest.TestCase):
    def test_missing_duplicate_and_reordered_trials_are_not_a_complete_suite(self):
        cases=evaluation_cases();check_case_list(cases,cases)
        for bad in (cases[:-1],cases[:-1]+[cases[1]],cases[::-1]):
            with self.assertRaises(ValueError):check_case_list(cases,bad)

    def test_physical_recording_must_match_perturbation_even_when_metadata_claims_zero_error(self):
        case=evaluation_cases()[1];base=np.array([1.,2.,.5,1.,0.,0.,0.]);velocity=np.array([1.,0.,0.,0.,0.,0.])
        joints=np.zeros(8);pose,vel=perturbed_root(base,velocity,case)
        trace=dict(root_pos=pose[None,:3].copy(),root_quat=pose[None,3:].copy(),environment_origin_w=np.zeros(3),
            root_com_vel=vel[None,:3].copy(),root_omega=vel[None,3:].copy(),joint_pos=joints[None].copy())
        reset=dict(case=case,maximum_initial_state_error=0.,commands_transformed=False,
            height_and_joint_pose_changed=False,initial_root_pose_w=pose.tolist(),initial_COM_velocity_w=vel.tolist())
        check_reset(trace,reset,case,base,velocity,joints,perturbed_root)
        trace['root_pos'][0,2]+=.001
        with self.assertRaisesRegex(ValueError,'initial state'):
            check_reset(trace,reset,case,base,velocity,joints,perturbed_root)

    def test_shared_reference_cannot_be_transformed_with_the_trial(self):
        case=evaluation_cases()[0];base=np.array([0.,0.,.5,1.,0.,0.,0.]);vel=np.array([1.,0.,0.,0.,0.,0.]);joints=np.zeros(8)
        trace=dict(root_pos=base[None,:3],root_quat=base[None,3:],environment_origin_w=np.zeros(3),
            root_com_vel=vel[None,:3],root_omega=vel[None,3:],joint_pos=joints[None])
        reset=dict(case=case,maximum_initial_state_error=0.,commands_transformed=True,height_and_joint_pose_changed=False,
            initial_root_pose_w=base.tolist(),initial_COM_velocity_w=vel.tolist())
        with self.assertRaisesRegex(ValueError,'Reference'):
            check_reset(trace,reset,case,base,vel,joints,perturbed_root)


if __name__=='__main__':unittest.main()
