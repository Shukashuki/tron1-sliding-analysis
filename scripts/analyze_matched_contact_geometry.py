"""Final matched analysis with actual XML collider offsets and orientations.

Run after the immutable SAC queue completes. The earlier link-centered envelope
is retained as a superseded diagnostic; training/evaluation are unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np


def descriptors(model):
    tree=ET.parse(model)
    result=[]
    for side in ('L','R'):
        geom=tree.find(f".//geom[@name='wheel_{side}_collision']")
        if geom is None or geom.get('type')!='cylinder':raise ValueError('Named cylinder required')
        size=np.fromstring(geom.get('size',''),sep=' ')
        pos=np.fromstring(geom.get('pos','0 0 0'),sep=' ')
        quat=np.fromstring(geom.get('quat','1 0 0 0'),sep=' ')
        if size.shape!=(2,) or pos.shape!=(3,) or quat.shape!=(4,) or not np.isclose(quat@quat,1.,atol=1e-8):
            raise ValueError('Invalid cylinder geometry')
        result.append(dict(name=geom.get('name'),radius_m=float(size[0]),half_width_m=float(size[1]),
                           local_center_m=pos.tolist(),local_quaternion_wxyz=quat.tolist()))
    return result


def cylinder_gaps(trace,geometry,rotate):
    columns=[]
    for side,geom in enumerate(geometry):
        q=trace['wheel_quaternion_wxyz'][:,side]
        center=trace['wheel_pos'][:,side]+rotate(q,np.asarray(geom['local_center_m']))
        local_axis=rotate(np.asarray(geom['local_quaternion_wxyz']),np.array([0.,0.,1.]))
        axis=rotate(q,local_axis);az=np.clip(np.abs(axis[:,2]),0,1)
        extent=geom['radius_m']*np.sqrt(np.maximum(0,1-az**2))+geom['half_width_m']*az
        columns.append(center[:,2]-extent)
    return np.stack(columns,axis=1)


def run(options,argv):
    if options.after_status:
        while True:
            try:prior=json.loads(options.after_status.read_text())
            except json.JSONDecodeError:
                time.sleep(2);continue
            state=prior['status']
            if state=='completed_with_analysis':break
            if 'stopped' in state or 'failed' in state:raise RuntimeError('Predecessor did not finish')
            time.sleep(30)
    actual=hashlib.sha256(options.base_analyzer.read_bytes()).hexdigest()
    if actual!=options.expected_base_sha256:raise ValueError('Frozen base analyzer changed')
    sys.path.insert(0,str(options.base_analyzer.parent))
    spec=importlib.util.spec_from_file_location('matched_base_analyzer',options.base_analyzer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('sac-root','ppo-root','validation-source','reference','model','legacy-source','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv);geometry=descriptors(args.model)
    def contact(trace,shared):
        gaps=cylinder_gaps(trace,geometry,shared.quaternion_rotate)
        t=trace['time_s'];select=(t>=1.44-1e-9)&(t<=4.+1e-9)
        g=gaps[select];dt=np.diff(t[select])
        def duty(v):return float(np.sum(v[:-1]*dt)/dt.sum())
        return dict(method='XML collider center and local axis transformed by measured wheel-link pose; cylinder envelope vs Z=0',
            geometry=geometry,native_force_threshold_used=False,
            exact_cooked_PhysX_geometry_equivalence_verified=False,
            core_gap_min_max_m=[float(g.min()),float(g.max())],
            core_any_penetration_over_5mm_fraction=duty((g<-.005).any(axis=1)),
            core_both_within_0p1mm_fraction=duty((g<=.0001).all(axis=1)),
            core_both_within_1mm_fraction=duty((g<=.001).all(axis=1)),
            core_both_within_2mm_fraction=duty((g<=.002).all(axis=1)))
    module.geometric_contact=contact
    module.run(args)
    path=args.output_dir/'report.json';report=json.loads(path.read_text())
    report['contact_geometry_correction']=dict(wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_analyzer_sha256=actual,model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        geometry=geometry,training_or_evaluation_modified=False,
        prior_link_centered_gap_superseded=True,physx_gap_remains_an_XML_envelope_proxy=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    prior=args.sac_root/'comparison_analysis'
    if prior.is_dir():
        (prior/'contact_geometry_superseded.json').write_text(json.dumps(dict(
            reason='Earlier envelope omitted collider-to-link center offsets; use corrected final analysis.',
            replacement_report=str(path),replacement_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps({'status':'completed_verified','final_report':str(path)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--after-status',type=Path)
    p.add_argument('--base-analyzer',type=Path,required=True)
    p.add_argument('--expected-base-sha256',required=True)
    options,argv=p.parse_known_args();run(options,argv)
