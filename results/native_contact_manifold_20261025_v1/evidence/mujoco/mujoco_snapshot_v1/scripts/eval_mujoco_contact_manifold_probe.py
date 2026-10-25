"""Counted observer in original K MuJoCo evaluation, with exact replay proof."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import mujoco_contact_manifold_probe as probe

BASE_SHA = 'cbb17fc4d2a7b1a4033fc8011222c8f2d99616ace4cba504c9e2537f53f5020c'
RECORDING_SHA = 'af790839128d5d02a8233a7a7b2cd8b23eb508e9056cd4d1c619b8d4173d14ef'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--base-snapshot', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    own, remaining = parser.parse_known_args(argv)
    base_file = own.base_snapshot.resolve()/'scripts/eval_ppo_ground_contact_v2k.py'
    assert sha(base_file) == BASE_SHA and sha(own.baseline_dir/'rollout.npz') == RECORDING_SHA
    sys.path.insert(1, str(base_file.parent))
    import eval_ppo_ground_contact_v2k as base
    assert Path(base.__file__).resolve() == base_file and probe.ACTIVE is None
    assert remaining.count('--backend') == 1 and remaining[remaining.index('--backend')+1] == 'mujoco'
    original_materialize, original_finish = base.native.materialize_source, base._finish
    sources = {str(Path(p).resolve()):sha(p) for p in (__file__, probe.__file__, probe.geometry.__file__, base_file)}
    def materialize(source, legacy_root):
        generated = original_materialize(source, legacy_root)
        anchor = '    def record(frame, torque):\n'
        assert generated.count(anchor) == 1
        generated = generated.replace(anchor, anchor+'        __import__("mujoco_contact_manifold_probe").observe(measured)\n', 1)
        compile(generated, '<K_native_contact_points>', 'exec')
        return generated
    def finish(report, raw, plan, *, contract=None):
        original_finish(report, raw, plan, contract=contract)
        assert probe.ACTIVE is not None
        audit = probe.ACTIVE.save(plan['output'], raw)
        assert sha(plan['output']/'rollout.npz') == RECORDING_SHA
        assert all(sha(p) == digest for p, digest in sources.items())
        assert plan['cohort']['input_sha256']['policy.pt'] == 'fafdccb45571b99da53a02455f2402a8cd8b960f16809ba8e6e0fe6996dce912'
        report['native_contact_manifold_probe'] = audit
        metadata = dict(kind='K_readonly_MuJoCo_native_contact_manifold_probe', status='completed_verified_requires_analysis',
            normalized_actor_sha256=plan['cohort']['input_sha256']['policy.pt'], physics_trajectory_sha256=RECORDING_SHA,
            read_only_baseline_byte_identical=True, source_sha256=sources, manifold_probe=audit,
            original_native_provenance=report['ground_contact_v2k_evaluation'], additional_training_updates=0,
            task_success_claimed=False)
        (plan['output']/'manifold_status.json').write_text(json.dumps(metadata, indent=2)+'\n')
    base.native.materialize_source, base._finish = materialize, finish
    try: base.run(remaining)
    finally: base.native.materialize_source, base._finish = original_materialize, original_finish


if __name__ == '__main__': main(sys.argv[1:])
