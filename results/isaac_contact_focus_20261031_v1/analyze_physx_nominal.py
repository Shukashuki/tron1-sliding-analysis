"""Reproduce the supplementary nominal PhysX SAC metrics without simulation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import analyze_ppo_backend_400hz_matrix as shared
from replay_physx_actions_mujoco import load_evaluator
from analyze_matched_contact_geometry import descriptors, cylinder_gaps
from analyze_nominal_contact_continuity import gap_metrics


def main(a):
    proof = json.loads((a.result_dir / 'nominal_status.json').read_text())
    recording = a.result_dir / 'nominal_physx_mu0p700/physics_trajectory.npz'
    assert shared.sha256(recording) == proof['recording_sha256']
    assert shared.sha256(a.model) == shared.INPUT_PINS['model']
    reference = shared.reference_data(a.reference)
    loader = load_evaluator(a.legacy_evaluator)
    model, _ = loader.load_model(a.model, no_visual_mesh=True)
    with np.load(recording, allow_pickle=False) as archive:
        trace, metrics = shared.analyze_case(dict(archive), 'physx', reference, shared.WheelKinematics(model))
    time = trace['time_s']
    select = (time >= 1.44 - 1e-9) & (time <= 4. + 1e-9)
    geometry = descriptors(a.model)
    gap = cylinder_gaps(trace, geometry, shared.quaternion_rotate)
    report = dict(kind='supplementary_PhysX_SAC_nominal_contact_focus',
        status='completed_descriptive_analysis', sample_size=1, mu=.7,
        training_backend='PhysX', evaluation_backend='PhysX',
        actor_sha256=proof['actor_sha256'], recording_sha256=proof['recording_sha256'],
        analyzer_sha256=shared.sha256(Path(__file__)), metrics=metrics,
        core_cylinder_envelope_proxy=gap_metrics(time[select], gap[select],
            trace['contact_count'][select], trace['contact_slip'][select]),
        geometry=geometry, model_sha256=shared.sha256(a.model), task_success_claimed=False,
        limitations=['XML envelope, not native cooked collision geometry',
            'No native per-point normal vectors in this recording; no friction-saturation certificate',
            'Same frozen turning task; not the hockey cohort', 'Single seed; no algorithm ranking'])
    (a.result_dir / 'nominal_descriptive_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Completed descriptive metrics; no task-success claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('result-dir', 'model', 'reference', 'legacy-evaluator'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
