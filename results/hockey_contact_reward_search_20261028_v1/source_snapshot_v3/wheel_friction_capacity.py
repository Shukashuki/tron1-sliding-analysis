"""Wheel-aggregate capacity indicators; not the solver's stick/slip flags."""
import numpy as np


def capacity_ratios(normal_force, tangent_force, touching, mu_static, mu_dynamic):
    if not np.isfinite([mu_static, mu_dynamic]).all() or not mu_static >= mu_dynamic > 0:
        raise ValueError('Require finite mu_static >= mu_dynamic > 0')
    fn=np.asarray(normal_force);ft=np.asarray(tangent_force);touch=np.asarray(touching,dtype=bool)
    if ft.shape != (*fn.shape,3) or touch.shape != fn.shape:
        raise ValueError('Contact-force shape mismatch')
    valid=touch & np.isfinite(fn) & (fn>1e-6) & np.isfinite(ft).all(-1)
    tangent=np.linalg.norm(ft[...,:2],axis=-1)
    ratios={}
    for key,mu in [('rho_static',mu_static),('rho_dynamic',mu_dynamic)]:
        value=np.full(fn.shape,np.nan)
        np.divide(tangent,mu*fn,out=value,where=valid)
        ratios[key]=value
    return dict(**ratios,valid=valid)
