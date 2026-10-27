#!/usr/bin/env python3
"""Two-mode single-world input intervention on the frozen hockey physical probe.

Target geometry, plant, actor weights, action mapping and actual state fields
remain fixed. Only documented reference-conditioning columns are replaced.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from hockey_conditioning_adapter import ConditioningAdapter,MODES


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('engine-scripts','prior-launch','bundle','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--mode',choices=MODES,required=True)
    args=p.parse_args()
    engine=args.engine_scripts/'generate_physx_policy_mppi_reference.py'
    if sha(engine)!='3aacb6b4f3b62d32ce6fa2326f2cb6bfe372b693b53c9830a27a85f26f3f7763':raise ValueError('Frozen engine changed')
    manifest=json.loads((args.bundle/'manifest.json').read_text())
    for row in manifest['files']:
        if sha(args.bundle/row['path'])!=row['sha256']:raise ValueError('Payload changed')
    old=json.loads(args.prior_launch.read_text())['argv'];get=lambda key:Path(old[old.index('--'+key)+1])
    with np.load(args.bundle/'nominal_conditioning.npz',allow_pickle=False) as z:
        conditioning=z['command'].copy();assert conditioning.shape==(139,14)
        assert int(z['original_reference_frames'])==349 and float(z['policy_dt_s'])==.02
    with np.load(args.bundle/'motion.npz',allow_pickle=False) as z:assert len(z['joint_pos'])==139
    sys.path.insert(0,str(args.engine_scripts))
    spec=importlib.util.spec_from_file_location('frozen_hockey_physics',engine)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import torch
    original_load=torch.jit.load;original_write=module.write;holder={}
    def load_actor(path,*pos,**kwargs):
        obj=original_load(path,*pos,**kwargs)
        if Path(path).resolve()==get('actor').resolve():
            if holder:raise RuntimeError('Unexpected second actor load')
            holder['adapter']=ConditioningAdapter(obj,conditioning,args.mode)
            return holder['adapter']
        return obj
    def write_report(path,value):
        if Path(path).name=='report.json':
            adapter=holder['adapter']
            if adapter.calls!=139 or value['steps']!=139 or value['worlds']!=1 or value['iterations']!=0:
                raise RuntimeError('Unexpected rollout scope')
            value.update(hockey_wrapper_sha256=sha(__file__),conditioning_adapter_sha256=sha(Path(__file__).with_name('hockey_conditioning_adapter.py')),
                conditioning_table_sha256=sha(args.bundle/'nominal_conditioning.npz'),conditioning_mode=args.mode,
                actual_actor_input_archived=True,reference_field_intervention_only=True,
                conditioning_original_reference_frames=349,hockey_target_reference_frames=139,
                hockey_target_unchanged=True,policy_weights_changed=False,MPPI_search_run=False,
                baseline_zero_residual=True,independent_hockey_evaluation_required=True,
                source_manifest_sha256=sha(args.bundle/'manifest.json'),
                qualification_target='unaltered hockey command; conditional-reference adapter is not itself posture imitation')
        original_write(path,value)
    torch.jit.load=load_actor;module.write=write_report
    a=argparse.Namespace(config=args.bundle/'config.json',actor=get('actor'),motion=args.bundle/'motion.npz',
        asset=get('asset'),original_motion=args.bundle/'motion.npz',seed_report=get('seed-report'),seed_trace=get('seed-trace'),
        tracking_root=get('tracking-root'),beyondmimic_root=get('beyondmimic-root'),output_dir=args.output_dir,
        device=args.device,steps=139,samples=None,iterations=0,validate_mean=args.bundle/'zero_mean.npz')
    try:module.run(a)
    finally:torch.jit.load=original_load


if __name__=='__main__':main()
