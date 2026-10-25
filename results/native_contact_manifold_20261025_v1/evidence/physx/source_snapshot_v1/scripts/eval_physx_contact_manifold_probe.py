"""Counted read-only point observer in the immutable authored-cylinder K evaluator."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import physx_contact_manifold_probe as probe

PARENT_SHA = '32b690dac706929b4df582a1b758fb0cbebe242f9ca8c5646f6c064d2a2eec62'
ACTOR_SHA = 'fafdccb45571b99da53a02455f2402a8cd8b960f16809ba8e6e0fe6996dce912'
BASELINE_SHA = 'cbd7d2b249737919bc21f2bccb05b3d82e8b869fbbe3d0a91c45e6b8eda4a67a'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    replacements = [
        ('        runtime.append(tire.TireRuntime(env,law=a.law,mu=.7))\n',
         '        runtime.append(tire.TireRuntime(env,law=a.law,mu=.7))\n        _point_install(env,geometry_audits)\n'),
        ('        report["wheel_contact_model_transfer"]=metadata;contract["wheel_contact_model_transfer"]=metadata\n',
         '        _point_finish(metadata,report,contract,current,observed,proof)\n'
         '        report["wheel_contact_model_transfer"]=metadata;contract["wheel_contact_model_transfer"]=metadata\n')]
    for old, new in replacements:
        assert source.count(old) == 1; source = source.replace(old, new, 1)
    compile(source, '<K_contact_point_observer>', 'exec')
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--contact-model-evaluator', type=Path, required=True)
    own, remaining = parser.parse_known_args(argv)
    assert sha(own.contact_model_evaluator) == PARENT_SHA
    for flag, value in (('--geometry', 'authored'), ('--law', 'native_coulomb'), ('--contact-offset-m', '.001'), ('--rest-offset-m', '0')):
        assert remaining.count(flag) == 1 and remaining[remaining.index(flag)+1] == value
    owners = []
    sources = {str(Path(p).resolve()):sha(p) for p in (__file__, probe.__file__, probe.geometry.__file__, own.contact_model_evaluator)}
    def install(env, rows):
        assert not owners; owners.append(probe.ContactManifoldProbe(env, rows))
    def finish(metadata, report, contract, plan, observed, proof):
        assert len(owners) == 1
        audit = owners[0].save(plan['output'])
        assert observed == proof['recording_sha256'] == BASELINE_SHA
        assert metadata['normalized_actor_sha256'] == ACTOR_SHA
        assert all(sha(p) == digest for p, digest in sources.items())
        metadata.update(kind='K_readonly_PhysX_native_contact_manifold_probe', manifold_probe=audit,
            additional_training_updates=0, source_sha256={**metadata['source_sha256'], **sources})
        report['native_contact_manifold_probe'] = audit; contract['native_contact_manifold_probe'] = audit
        (plan['output']/'manifold_status.json').write_text(json.dumps(metadata, indent=2)+'\n')
    namespace = dict(__name__='_K_contact_point_observer', __file__=str(own.contact_model_evaluator.resolve()),
        _point_install=install, _point_finish=finish)
    old = sys.argv
    try:
        sys.path.append(str(own.contact_model_evaluator.resolve().parent))
        sys.argv = [str(own.contact_model_evaluator), *remaining]
        exec(compile(materialize(own.contact_model_evaluator.read_text()), '<K_contact_point_observer>', 'exec'), namespace)
        namespace['main']()
    finally: sys.argv = old


if __name__ == '__main__': main(sys.argv[1:])
