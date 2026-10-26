"""Independent same-world repeatability and1.5x MPPI rescue audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare_physx_speed_references import load
from analyze_physx_speed_ladder import case_metrics
from analyze_physx_policy_mppi_reference import verify_native_buffers


def run(a):
    assert not a.output.exists()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    status=json.loads((a.rescue/'status.json').read_text());assert status['status']=='completed_search_and_fresh_single_world_validation'
    search=json.loads((a.rescue/'search_v1/report.json').read_text())
    final=json.loads((a.rescue/'validation_v1/report.json').read_text())
    for directory,report in [(a.rescue/'search_v1',search),(a.rescue/'validation_v1',final)]:
        assert report['worlds']==1 and all(sha(directory/k)==v for k,v in report['artifacts'].items())
    np.testing.assert_array_equal(load(a.rescue/'search_v1/mean.npz')['latent_mean'],load(a.rescue/'validation_v1/mean.npz')['latent_mean'])
    trace=load(a.rescue/'validation_v1/final_rollout.npz');pred=load(a.rescue/'search_v1/final_rollout.npz')
    baseline=load(a.baseline/'final_rollout.npz');searchbase=load(a.rescue/'search_v1/baseline.npz')
    signals=('root_pos','root_quat','root_com_vel','root_omega','joint_pos','joint_vel','normal_fz','friction_force_w','friction_anchor_power')
    repeat_errors={k:float(np.max(abs(trace[k]-pred[k]))) for k in signals}
    baseline_errors={k:float(np.max(abs(baseline[k]-searchbase[k]))) for k in signals}
    reference=load(a.reference);scale=search['initial_velocity_multiplier']
    metrics=case_metrics(trace,reference,scale)
    raw=verify_native_buffers(trace,a.rescue/'validation_v1/native_contact_buffers.npz')
    history=search['search_history'];baseline_cost=history[0]['costs'][0];final_cost=final['final_costs'][0]
    result=dict(kind='serial_single_world_PhysX_MPPI_speed_rescue',requested_scale=1.5,effective_scale=scale,
        baseline=case_metrics(baseline,reference,scale),candidate=metrics,
        baseline_cost=baseline_cost,final_mean_cost=final_cost,relative_cost_change=final_cost/baseline_cost-1.,
        baseline_full_vs_light_cost_difference=history[1]['cost_difference_from_full_baseline'],
        baseline_probe_vs_search_baseline_max_errors=baseline_errors,
        search_final_mean_vs_fresh_validation_max_errors=repeat_errors,
        exact_baseline_and_final_state_reproduction=all(v==0 for v in [*repeat_errors.values(),*baseline_errors.values()]),
        native_contact_verification=raw,MPPI_importance_effective_sample_sizes=[h['importance']['ess'] for h in history if 'importance' in h],
        same_400Hz_cost_in_full_and_light_modes=True,
        planning_mode='single world; sequential complete rollouts; normal force matrix readback verified against contact points; telemetry omitted for proposals only',
        policy_training=False,new_controller_or_reward_baseline_claimed=False,
        best_sample_substituted_for_final_mean=False,task_success_claimed=False,
        interpretation='Small-sample weighted mean did not rescue1.5x tracking; worse final cost and core XY. Identical repeats establish computational repeatability for this case only, not a robustness certificate.',
        source_sha256=sha(__file__),search_report_sha256=sha(a.rescue/'search_v1/report.json'),
        validation_report_sha256=sha(a.rescue/'validation_v1/report.json'))
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('rescue','baseline','reference','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
