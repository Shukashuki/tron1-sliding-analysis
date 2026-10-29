"""Audit reference-phase geometry under the frozen RSI sampling rule.

This is an analytic phase-selection calculation, not observed reset contacts.
No simulator, policy, reference correction or training job is invoked.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bin_geometry_probabilities(bilateral, bins):
    """Integrate floor((bin + U[0,1)) * (frames - 1) / bins) exactly."""
    bilateral = np.asarray(bilateral, dtype=bool)
    if bilateral.ndim != 1 or len(bilateral) < 2 or bins < 1:
        raise ValueError('Invalid reference or bin count')
    width = (len(bilateral) - 1) / bins
    edges = np.arange(bins + 1, dtype=float) * width
    # The terminal frame has zero probability in the continuous sampling rule.
    frames = np.arange(len(bilateral) - 1)
    overlap = np.maximum(0., np.minimum(edges[1:, None], frames + 1)
                         - np.maximum(edges[:-1, None], frames))
    np.testing.assert_allclose(overlap.sum(1), width, rtol=0, atol=1e-12)
    return overlap @ bilateral[:-1].astype(float) / width


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'geometry', 'reference', 'repository', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    read = lambda path: json.loads(path.read_text())
    manifest = read(a.bundle / 'manifest.json')
    for row in manifest['files']:
        if sha(a.bundle / row['path']) != row['sha256']:
            raise ValueError('Frozen source changed: ' + row['path'])
    upstream = read(a.bundle / 'upstream_provenance.json')
    tracking = 'upstream/tron1-learning-from-human/training/tron1_tracking.py'
    commands = ('upstream/whole_body_tracking/source/whole_body_tracking/'
                'whole_body_tracking/tasks/tracking/mdp/commands.py')
    for name in (tracking, commands):
        if sha(a.repository / name) != upstream[name]:
            raise ValueError('Upstream reset source changed')
    calls = [n for n in ast.walk(ast.parse((a.repository / tracking).read_text()))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'TronMotionCommandCfg']
    if len(calls) != 1:
        raise ValueError('Ambiguous motion configuration')
    keywords = {k.arg: k.value for k in calls[0].keywords}
    bin_seconds = ast.literal_eval(keywords['sampling_bin_seconds'])
    reset_noise = {k: ast.literal_eval(keywords[k]) for k in
                   ('pose_range', 'velocity_range', 'joint_position_range')}
    plan = read(a.bundle / 'plan.json')
    configs = [read(a.bundle / row['protocol']) for row in plan['cells']]
    settings = {(c['policy_dt_s'], c['rsi_full_clip_start_fraction'],
                 c['adaptive_sampling']) for c in configs}
    if len(settings) != 1:
        raise ValueError('Cells do not share an RSI configuration')
    dt, full, sampler = settings.pop()
    if sampler != 'fall_only' or not 0 <= full <= 1:
        raise ValueError('Unsupported sampler')
    geometry = read(a.geometry / 'report.json')
    ref_sha = sha(a.reference / 'motion.npz')
    if ref_sha != geometry['input_sha256']['reference'] or ref_sha != plan['reference_sha256']['motion.npz']:
        raise ValueError('Reference geometry is from another reference')
    with np.load(a.geometry / 'geometry_diagnostics.npz', allow_pickle=False) as z:
        gap = z['reference_gap_m']
        times = z['reference_time_s']
    with np.load(a.reference / 'motion.npz', allow_pickle=False) as z:
        if len(z['joint_pos']) != len(gap) or not np.isclose(float(z['fps']), 1 / dt):
            raise ValueError('Reference clock mismatch')
    np.testing.assert_allclose(times, np.arange(len(gap)) * dt, atol=1e-6, rtol=0)
    tolerance = configs[0]['bilateral_sliding']['contact_tolerance_m']
    if any(c['bilateral_sliding']['contact_tolerance_m'] != tolerance for c in configs):
        raise ValueError('Contact tolerance differs across cells')
    bilateral = (gap <= tolerance).all(-1)
    bins = max(2, math.ceil(len(gap) * dt / bin_seconds))
    probability = bin_geometry_probabilities(bilateral, bins)
    fixed = full * float(bilateral[0])
    report = dict(
        kind='hockey_rsi_reference_geometry_v1',
        source_manifest_sha256=sha(a.bundle / 'manifest.json'),
        script_sha256=sha(Path(__file__)), reference_sha256=ref_sha,
        geometry_report_sha256=sha(a.geometry / 'report.json'),
        geometry_arrays_sha256=sha(a.geometry / 'geometry_diagnostics.npz'),
        upstream_reset_sources={k: upstream[k] for k in (tracking, commands)},
        sampler_sha256=sha(a.bundle / 'hockey_ppo_sampling.py'),
        frames=len(gap), bilateral_reference_frames=np.flatnonzero(bilateral).tolist(),
        contact_tolerance_m=tolerance, policy_dt_s=dt,
        sampling_bin_seconds=bin_seconds, bin_count=bins,
        full_clip_start_fraction=full, configured_reset_noise=reset_noise,
        per_bin_bilateral_geometry_probability=probability.tolist(),
        uniform_bin_counterfactual_bilateral_probability=float(fixed + (1-full)*probability.mean()),
        arbitrary_bin_weight_bounds=[float(fixed+(1-full)*probability.min()),
                                    float(fixed+(1-full)*probability.max())],
        measured_training_reset_contact_fraction=None,
        limitations=[
            'The actual adaptive bin weights and reset contacts were not archived.',
            'Probabilities describe selected reference geometry before physics; joint-limit clipping is not simulated.',
            'Finite-precision PRNG endpoint effects are excluded from the analytic continuous-uniform rule.',
            'Native contact, loading, causality and task success cannot be inferred from this calculation.',
            'No reference, reward, active training snapshot or success gate was modified.'])
    a.output.mkdir(parents=True)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('bin_count', 'bilateral_reference_frames',
          'uniform_bin_counterfactual_bilateral_probability', 'arbitrary_bin_weight_bounds')}))


if __name__ == '__main__':
    main()
