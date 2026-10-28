"""CPU replay of recorded PPO inputs and an optional saved MPPI residual plan.

This checks the origin of applied actions, not dynamics or task success. The
residual is reconstructed independently with SciPy's cubic Hermite interpolator.
The actor applies its saved normalizer internally, exactly once.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from evaluate_hockey_physx import (audit_actor_observations, load, q_matrix, sha,
                                   verify_hockey_search_source)

ACTOR_SHA = 'e5b94e87ed070f28c391cc7b0a11b216ce6b6d887141a28e7c83157085a9d8d0'
TOLERANCE = 5e-5


def reconstruct_residual(mean, reference_frames, control_steps):
    mean = np.asarray(mean, dtype=float)
    if (mean.ndim != 2 or mean.shape[1] != 8 or not 2 <= len(mean) <= reference_frames
            or not np.isfinite(mean).all() or not 1 <= control_steps <= reference_frames):
        raise ValueError('Invalid mean or reference/control clock')
    times = np.arange(control_steps)*.02
    # The smoke check uses the first 16 samples of the FULL reference basis.
    # Compressing the 139-frame spline into 16 frames would change the controller.
    knots = np.linspace(0, (reference_frames-1)*.02, len(mean))
    values = CubicHermiteSpline(knots, mean, np.zeros_like(mean), axis=0)(times)
    ramp = np.clip(times/.10, 0, 1)
    return values*(ramp*ramp*(3-2*ramp))[:, None]


def replay_actions(actor, observations, recorded, residual=None):
    import torch
    observations = np.asarray(observations)
    recorded = np.asarray(recorded)
    if (observations.ndim != 2 or observations.shape[1] != 51
            or recorded.shape != (len(observations), 8)
            or not np.isfinite(observations).all() or not np.isfinite(recorded).all()
            or np.any(abs(recorded)>1+1e-6)):
        raise ValueError('Invalid recorded action/input array')
    with torch.inference_mode():
        raw = actor(torch.tensor(observations, dtype=torch.float32)).cpu().numpy()
    if raw.shape != recorded.shape or not np.isfinite(raw).all():
        raise ValueError('Invalid actor output')
    prior = np.clip(raw, -1, 1)
    if residual is None:
        residual = np.zeros_like(prior)
    if np.shape(residual) != prior.shape or not np.isfinite(residual).all():
        raise ValueError('Invalid decoded residual')
    decoded = np.clip(prior+residual, -1, 1)
    error = abs(decoded-recorded)
    if np.max(error, initial=0)>TOLERANCE:
        raise ValueError(f'Applied action replay mismatch: {error.max():.9g}')
    return dict(raw_actor=raw, bounded_prior=prior, residual=residual,
                decoded_actions=decoded, maximum_error=float(error.max(initial=0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--actor', type=Path, required=True)
    p.add_argument('--command-dir', type=Path, required=True)
    p.add_argument('--conditioning-table', type=Path)
    p.add_argument('--search-source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    run = json.loads((a.run_dir/'report.json').read_text())
    trace_path = a.run_dir/'final_rollout.npz'
    if sha(a.actor) != ACTOR_SHA or run['actor_sha256'] != ACTOR_SHA:
        raise ValueError('Wrong frozen PPO actor')
    if sha(trace_path) != run['artifacts']['final_rollout.npz']:
        raise ValueError('Recorded trajectory changed')
    motion = a.command_dir/'motion.npz'
    if sha(motion) != run['motion_sha256']:
        raise ValueError('Wrong physical target')
    trace = load(trace_path);reference = load(motion)
    steps = len(trace['actions'])
    if len(trace['root_pos']) != steps*8+1:
        raise ValueError('Action/state clock mismatch')
    mode = run.get('conditioning_mode')
    conditioning = None
    if mode is not None:
        if a.conditioning_table is None or sha(a.conditioning_table) != run['conditioning_table_sha256']:
            raise ValueError('Wrong conditioning table')
        conditioning = load(a.conditioning_table)['command'][:steps]
    elif a.conditioning_table is not None:
        raise ValueError('Unexpected conditioning table')
    input_audit = audit_actor_observations(trace, reference, conditioning, mode)
    plan_sha = None;source_audit = None;residual = None
    if run['kind'] == 'hockey_native_physx_mppi_v1':
        if a.search_source is None:
            raise ValueError('Native MPPI source snapshot is required')
        source_audit = verify_hockey_search_source(a.search_source, run)
        plan = a.run_dir/'mean.npz';plan_sha = sha(plan)
        if plan_sha != run['artifacts']['mean.npz']:
            raise ValueError('Saved MPPI plan changed')
        residual = reconstruct_residual(load(plan)['latent_mean'], len(reference['joint_pos']), steps)
    else:
        if a.search_source is not None or run.get('iterations') != 0 or run.get('search_history'):
            raise ValueError('Unsupported historical controller')
        if np.any(load(a.run_dir/'mean.npz')['latent_mean'] != 0):
            raise ValueError('Historical probe is not zero residual')
        if sha(a.run_dir/'mean.npz') != run['artifacts']['mean.npz']:
            raise ValueError('Historical residual artifact changed')
    import torch
    torch.set_num_threads(2)
    actor = torch.jit.load(str(a.actor), map_location='cpu').eval()
    result = replay_actions(actor, trace['observations'], trace['actions'], residual)
    controlled_tilt = np.degrees(np.arccos(np.clip(q_matrix(trace['root_quat'][np.arange(steps)*8])[:,2,2], -1, 1)))
    first = np.flatnonzero(controlled_tilt >= 45)
    before = np.arange(steps) < (first[0] if len(first) else steps)
    report = dict(kind='hockey_recorded_action_replay_v1', status='passed',
        audit_script_sha256=sha(__file__), actor_sha256=ACTOR_SHA,
        run_report_sha256=sha(a.run_dir/'report.json'), trace_sha256=sha(trace_path),
        motion_sha256=sha(motion), plan_sha256=plan_sha, source_audit=source_audit,
        actor_input_audit=input_audit, conditioning_mode=mode, control_steps_checked=steps,
        maximum_action_error=result['maximum_error'], tolerance=TOLERANCE,
        raw_actor_max_abs=float(abs(result['raw_actor']).max()),
        clipped_actor_element_fraction=float(np.mean(abs(result['raw_actor'])>1)),
        pre_45deg_control_sample_count=int(before.sum()),
        pre_45deg_actor_clipped_element_fraction=float(np.mean(abs(result['raw_actor'][before])>1)) if before.any() else None,
        residual_max_abs=float(abs(result['residual']).max()),
        normalizer='Applied exactly once inside the frozen actor; raw 51D observations are passed unchanged.',
        residual_reconstruction='Independent SciPy CubicHermiteSpline with zero knot derivatives and initial 100ms ramp' if residual is not None else 'verified zero mean',
        physics_run=False, task_success_claimed=False, torch_version=torch.__version__)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'replay.npz', time_s=np.arange(steps)*.02,
        **{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    report['replay_sha256']=sha(a.output/'replay.npz')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','control_steps_checked','maximum_action_error','clipped_actor_element_fraction','physics_run')},indent=2))


if __name__ == '__main__':
    main()
