"""Use deterministic cylinder support-projection separation in the optimizer.

Positive separation is a conservative certificate of disjoint collision
cylinders,not an exact Euclidean signed distance. This replaces the unstable
geometry-distance query only;contact physics and planning costs stay fixed.
"""
from __future__ import annotations
import argparse
import inspect
import textwrap
import hashlib
from pathlib import Path

import numpy as np
import generate_camber_reference as base
import hybrid_contact_reference as parent
from hybrid_contact_reference_support import SupportProblem

PARENT_SHA='4fac92622148c6cad33ee4aef83423746517c81ee86f5e193d76705863db14eb'
SUPPORT_SHA='37feb5277f8d441923b9e0753ddf77000801b3a1713bfd6bc42ee29097689518'


def _certificates(center_a,axis_a,size_a,center_b,axis_b,size_b):
    a=axis_a/np.linalg.norm(axis_a,axis=-1,keepdims=True)
    b=axis_b/np.linalg.norm(axis_b,axis=-1,keepdims=True);delta=center_b-center_a
    directions=np.stack([a,b,delta,delta-a*np.sum(a*delta,axis=-1,keepdims=True),
                         delta-b*np.sum(b*delta,axis=-1,keepdims=True)],axis=1)
    length=np.linalg.norm(directions,axis=-1,keepdims=True)
    n=np.divide(directions,length,out=np.zeros_like(directions),where=length>1e-12)
    support_a=size_a[:,1,None]*abs(np.sum(a[:,None]*n,axis=-1))+size_a[:,0,None]*np.linalg.norm(np.cross(a[:,None],n),axis=-1)
    support_b=size_b[:,1,None]*abs(np.sum(b[:,None]*n,axis=-1))+size_b[:,0,None]*np.linalg.norm(np.cross(b[:,None],n),axis=-1)
    gaps=abs(np.sum(delta[:,None]*n,axis=-1))-support_a-support_b
    gaps=np.where(length[:,:,0]>1e-12,gaps,-np.inf)
    return gaps.max(-1)


def cylinder_separation_certificate(center_a,axis_a,size_a,center_b,axis_b,size_b):
    args=[np.asarray(a,dtype=float)[None] for a in (center_a,axis_a,size_a,center_b,axis_b,size_b)]
    return float(_certificates(*args)[0])


def opposite_leg_certificate(model,data,groups):
    left=np.repeat(groups[0],len(groups[1]));right=np.tile(groups[1],len(groups[0]))
    return float(_certificates(data.geom_xpos[left],data.geom_xmat[left].reshape(-1,3,3)[:,:,2],model.geom_size[left,:2],
        data.geom_xpos[right],data.geom_xmat[right].reshape(-1,3,3)[:,:,2],model.geom_size[right,:2]).min())


def materialize_evaluate():
    assert base.sha(parent.__file__)==PARENT_SHA
    source=textwrap.dedent(inspect.getsource(parent.Problem.evaluate))
    old='opposite=min(mj.mj_geomDistance(m,d,left,right,1.,None) for left in self.opposite[0] for right in self.opposite[1])'
    assert source.count(old)==1;source=source.replace(old,'opposite=opposite_leg_certificate(m,d,self.opposite)',1)
    ns=dict(np=np,dc_bounds=parent.dc_bounds,cylinder_support=parent.cylinder_support,rim_points=parent.rim_points,
            RESIDUAL_DIM=parent.RESIDUAL_DIM,opposite_leg_certificate=opposite_leg_certificate)
    exec(compile(source,'<support_projection_evaluate>','exec'),ns)
    return ns['evaluate'],hashlib.sha256(source.encode()).hexdigest()


_evaluate,MATERIALIZED_EVALUATE_SHA=materialize_evaluate()


class StableSupportProblem(SupportProblem):
    evaluate=_evaluate


def run(a):
    source_path=Path(__file__).with_name('hybrid_contact_reference_support.py')
    assert base.sha(source_path)==SUPPORT_SHA
    source=source_path.read_text()
    changes=[('p=SupportProblem(mujoco.MjModel','p=_StableSupportProblem(mujoco.MjModel'),
        ("parent_source_sha256=base.sha(Path(__file__).with_name('hybrid_contact_reference.py')),",
         "parent_source_sha256=base.sha(Path(__file__).with_name('hybrid_contact_reference.py')),\n"
         "        support_template_sha256='"+SUPPORT_SHA+"',\n"
         "        separation_certificate='cylinder support projection lower bound;positive implies disjoint',\n"
         "        materialized_evaluate_sha256='"+MATERIALIZED_EVALUATE_SHA+"',")]
    for old,new in changes:
        assert source.count(old)==1;source=source.replace(old,new,1)
    ns=dict(__name__='_stable_support_driver',__file__=str(Path(__file__).resolve()),_StableSupportProblem=StableSupportProblem)
    exec(compile(source,'<stable_support_driver>','exec'),ns);ns['run'](a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('reference','model','initial-fit','config','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
