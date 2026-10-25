"""Force-only offline evidence for P; never predict learning from stored poses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

import analyze_physx_normal_compliance as native
import ground_contact_reward_v2p as definition


def run(args):
    assert not args.output_dir.exists()
    native.verify_collection(args.collection)
    rows = []
    saved = {}
    for key in native.ORDER:
        folder = args.collection/'turn_models_v2'/key
        metadata = native.read(folder/'normal_compliance_status.json')
        assert native.sha(folder/'physics_trajectory.npz') == metadata['physics_trajectory_sha256']
        raw = native.npz(folder/'physics_trajectory.npz')
        force_all = raw['wheel_ground_contact_force_w_n'][:, 0, :, 2]
        mass = metadata['normal_contact_model_intervention']['robot_native_mass_kg']
        force = force_all[1:].reshape(349, 8, 2)
        times = np.arange(349)*.02
        assert np.array_equal(raw['reference_frame_for_action'][1:, 0].reshape(349, 8), np.broadcast_to(np.arange(349)[:, None], (349, 8)))
        values = definition.reward_numpy(force, mass*9.81, 1., 1., 1., 1.)
        gap = (force.max(-1) <= 5).any(-1)
        impact = values['ungated_peak_impact_squared_excess'] > 0
        core = (times >= 1.44-1e-9) & (times < 4.-1e-9)
        peak_idx = np.unravel_index(force_all.argmax(), force_all.shape)
        rows.append(dict(id=key, policy_windows=349, gap_windows=int(gap.sum()),
            over3mg_windows=int(impact.sum()), over3mg_and_gap_windows=int((gap & impact).sum()),
            core_gap_windows=int((gap & core).sum()), core_over3mg_windows=int((impact & core).sum()),
            positive_impact_gate_median_in_over3mg_windows=float(np.median(values['impact_gate'][impact])),
            new_ungated_cost_median_in_over3mg_windows=float(np.median(values['ungated_peak_impact_cost'][impact])),
            whole_recording_peak_force_n=float(force_all.max()), whole_recording_peak_time_s=float(raw['time_s'][peak_idx[0]]),
            policy_controlled_peak_force_n=float(force.max()),
            core_peak_force_n=float(force[core].max()), actual_weight_n=mass*9.81,
            physics_trajectory_sha256=metadata['physics_trajectory_sha256']))
        saved.update({key+'__'+name: value for name, value in dict(policy_time_s=times,
            forces_n=force, positive_impact_gate=values['impact_gate'],
            new_ungated_peak_impact_cost=values['ungated_peak_impact_cost'], gap=gap).items()})
    result = dict(kind='offline_force_window_peak_impact_cost_diagnostic', rows=rows,
        initial_native_sample_excluded_from_policy_reward_windows=True,
        tracking_speed_beta_qualities_set_to_one_only_for_isolated_force_arithmetic=True,
        actual_training_rewards_reconstructed=False, stored_poses_actions_unchanged=True,
        physical_models_unchanged=True, new_P_training_or_task_success_claimed=False,
        no_causal_or_learning_outcome_from_offline_association=True,
        new_definition_sha256=native.sha(definition.__file__), diagnostic_source_sha256=native.sha(__file__))
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir/'force_windows.npz', **saved)
    (args.output_dir/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    run(p.parse_args())
