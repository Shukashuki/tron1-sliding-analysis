"""Native leg-motion diagnostic, excluding wheel spin by actual joint names.

This is offline evidence for the next intervention, not a new reward, gate or
claim that leg velocity causes the observed root chatter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_contact_taskspace_margin as analysis

LEGS = ["abad_L_Joint","abad_R_Joint","hip_L_Joint","hip_R_Joint","knee_L_Joint","knee_R_Joint"]


def band_rms(value, dt):
    n = len(value)
    a = np.stack((np.linspace(-1.,1.,n),np.ones(n)),axis=-1)
    residual = value-a@np.linalg.lstsq(a,value,rcond=None)[0]
    w = np.hanning(n)
    transformed = np.fft.rfft(residual*w[:,None],axis=0)
    factor = np.full(len(transformed),2.);factor[0] = 1.
    if n%2 == 0:factor[-1] = 1.
    power = abs(transformed)**2*factor[:,None]/(n*np.sum(w*w))
    np.testing.assert_allclose(power.sum(0),(residual**2*w[:,None]**2).sum(0)/np.sum(w*w),rtol=1e-12,atol=1e-12)
    frequency = np.fft.rfftfreq(n,dt)
    mask = (frequency>10.) & (frequency<100.)
    return float(np.sqrt(power[mask].sum(0).mean()))


def run(root):
    root = Path(root)
    out = root/"leg_motion_diagnostic_v1.json"
    assert not out.exists()
    d = analysis.read(root/"analysis/analysis.json")
    cases = {}
    for row in d["rows"]:
        engine = row["engine"]
        folder = (analysis.CONTROL if row["variant"]=="control" else root/"completed_evidence_v1")
        prefix = "K" if row["variant"]=="control" else "M"
        p = folder/"nominal_v1"/(prefix+"_"+engine)/("physics_trajectory.npz" if engine=="physx" else "rollout.npz")
        assert analysis.sha(p) == row["recording_sha256"]
        with np.load(p,allow_pickle=False) as a:
            names = a["joint_names"].astype(str).tolist()
            assert len(names)==len(set(names))==8
            ids = [names.index(n) for n in LEGS]
            assert not any(names[i].startswith("wheel_") for i in ids)
            velocity = a["joint_velocity_rad_s"] if engine=="physx" else a["joint_vel"]
            velocity = np.asarray(velocity,dtype=np.float64).reshape(-1,8)[:,ids]
            time = a["time_s"].reshape(-1)
        select = (time>=1.44-1e-12) & (time<=4.+1e-12)
        v = velocity[select]
        assert len(v)==1025 and np.isfinite(v).all()
        np.testing.assert_allclose(np.diff(time[select]),.0025,rtol=0,atol=1e-12)
        cases[row["id"]] = dict(joint_names=LEGS,native_core_samples=len(v),sample_dt_s=.0025,
            actual_leg_velocity_channel_mean_RMS_rad_s=float(np.sqrt(np.mean(v*v))),
            actual_leg_velocity_RMS_per_joint_rad_s=np.sqrt(np.mean(v*v,axis=0)).tolist(),
            actual_leg_velocity_10_to_100Hz_channel_mean_RMS_rad_s=band_rms(v,.0025),
            finite_difference_leg_acceleration_channel_mean_RMS_rad_s2=float(np.sqrt(np.mean((np.diff(v,axis=0)/.0025)**2))),
            root_angular_XY_HF_ratio_to_original_same_engine_PPO=row["nominal_screen"]["core_HF_ratio_to_original_same_engine_PPO"],
            recording_sha256=analysis.sha(p))
    result = dict(kind="native_leg_motion_not_wheel_spin",cases=cases,source_sha256=analysis.sha(__file__),
        analysis_sha256=analysis.sha(root/"analysis/analysis.json"),
        spectral_definition="Linear detrend, Hann, one-sided Parseval channel-mean power, 10<f<100Hz; full-spectrum Parseval independently checked",
        root_motion_causally_attributed_to_legs=False,new_reward_or_acceptance_installed=False,task_success_claimed=False)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(out),"task_success_claimed":False}))


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--result",type=Path,required=True)
    run(p.parse_args().result)
