"""Actor-free replay of each matched400 PPO origin's nominal PhysX actions.

The old diagnostic implementation is retained; only explicitly bound new
episode hashes and the shared wheel-priority material mapping are supplied.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import replay_physx_actions_mujoco as replay
import analyze_physx_mujoco_request_replay as analysis


CASES={
    'physx':dict(physx_physics='0194cde85053a24963f19ca9e853aefe3b21ac2f18c62d9442606467b0910b98',
        physx_report='893453d8c004eec65eb4d291c47f6172d2b872b05b08e17f169459ef719201fa',
        actor='ab3c486ef779deb0cff52bfb33415f5c3a779381740a2c12898506e4e80f08bc'),
    'mjlab':dict(physx_physics='1f6d73be39c626d50755b609f3dcc60a951c41af65d0e70d83648bd49ff4856c',
        physx_report='435d1a17d8c196deb9836d1fbd9f022ce9058f3e65d6e00386b4b2f92aa8cdaa',
        actor='081062a886db35740a68a6cead4579cd339cff62c311728c2f6ce71ed5a8c5f0')}


def run(a):
    expected=CASES[a.training_backend]
    if replay.sha256(Path(replay.__file__))!='22a6d025b6fb7e1159a5bdf611a5eac876cc072b12afa479f2217e167fbd1096':
        raise ValueError('Frozen request-replay implementation differs')
    report=json.loads(a.physx_report.read_text())
    audit=report['backend_400hz_cohort_evaluation']
    assert audit['status']=='verified' and audit['training_backend']==a.training_backend and not audit['smoke_only']
    assert audit['normalized_actor_sha256']==expected['actor']
    replay.INPUT_PINS={**replay.INPUT_PINS,**{k:expected[k] for k in ('physx_physics','physx_report')}}
    original=replay.load_evaluator
    material_proof={}
    patched=[]
    def load(path):
        module=original(path)
        from training import tron1_drift
        configure=tron1_drift.configure_mujoco_materials
        def material(model,profile):
            result=configure(model,profile)
            assert model.npair==0
            colliding=np.flatnonzero((model.geom_contype!=0)|(model.geom_conaffinity!=0))
            wheels=[]
            for index in colliding:
                wheel=model.body(int(model.geom_bodyid[index])).name in ('wheel_L_Link','wheel_R_Link')
                model.geom_priority[index]=int(wheel)
                if wheel:wheels.append(int(index))
            assert len(wheels)>=2 and np.allclose(model.geom_friction[colliding,0],.7,rtol=0,atol=1e-12)
            material_proof.update(colliding_geom_ids=colliding.tolist(),wheel_geom_ids=wheels,
                geom_priority=model.geom_priority[colliding].tolist(),sliding_mu=model.geom_friction[colliding,0].tolist(),
                explicit_contact_pairs=int(model.npair),physics_dt_s=float(model.opt.timestep))
            result['matched_backend_priority_audit']=dict(material_proof)
            return result
        patched.append((tron1_drift,configure))
        tron1_drift.configure_mujoco_materials=material
        return module
    replay.load_evaluator=load
    try:
        result=replay.run(a)
    finally:
        replay.load_evaluator=original
        for module,configure in patched:module.configure_mujoco_materials=configure
    result.update(matched_training_backend=a.training_backend,matched_actor_sha256=expected['actor'],
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),material_mapping=material_proof,
        transfer_direction='same PhysX requests replayed in CPU MuJoCo; no inverse replay claimed')
    (a.output_dir/'request_replay_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    return analysis.run(argparse.Namespace(replay_dir=a.output_dir,output_dir=a.output_dir/'analysis'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-evaluator','physx-physics','physx-report','motion','model','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-backend',choices=tuple(CASES),required=True)
    p.add_argument('--max-policy-steps',type=int,default=349)
    a=p.parse_args();print(json.dumps(run(a),indent=2))
