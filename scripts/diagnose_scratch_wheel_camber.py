"""Wheel camber from verified native replay poses; geometry only,no stepping."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def run(args):
    import mujoco
    root = args.result; output = root/"wheel_camber_diagnostic_v1.json"
    assert not output.exists()
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    proof = read(root/"artifact_verification_v1.json"); provenance = read(root/"render/provenance.json")
    assert proof["status"] == "verified" and proof["native_numerical_recomputation_matches"]
    assert proof["render_provenance_sha256"] == sha(root/"render/provenance.json")
    assert sha(args.model) == provenance["source_model_sha256"]
    states_path = root/"render/replay_states.npz"
    assert sha(states_path) == provenance["artifacts"]["replay_states.npz"]
    model = mujoco.MjModel.from_xml_path(str(args.model)); data = mujoco.MjData(model)
    wheels = [model.body(n).id for n in ("wheel_L_Link", "wheel_R_Link")]
    rows = {}
    with np.load(states_path, allow_pickle=False) as states:
        for key in provenance["case_order"]:
            q = states[key+"__qpos"]; force = states[key+"__floor_normal_force_w"]
            time = np.arange(len(q))*.0025
            core = (time >= 1.44-1e-9) & (time <= 4.+1e-9)
            angles = []
            for pose in q:
                data.qpos[:] = pose
                mujoco.mj_kinematics(model, data)
                # Wheel local+Y is its axle.0deg=vertical wheel;90deg=flat wheel.
                axes = data.xmat[wheels].reshape(2,3,3)[:,:,1]
                angles.append(np.degrees(np.arcsin(abs(axes[:,2]).clip(0.,1.))))
                assert data.time == 0.
            angles = np.asarray(angles); row = {}
            for i, name in enumerate(("left", "right")):
                selected = core & (force[:,i,2] > 5.)
                a = angles[selected,i]
                row[name] = dict(core_loaded_samples=int(selected.sum()),
                    loaded_camber_deg_median=None if not len(a) else float(np.median(a)),
                    loaded_camber_deg_p95=None if not len(a) else float(np.quantile(a,.95)),
                    core_max_camber_deg=float(angles[core,i].max()),
                    loaded_fraction_camber_above60deg=None if not len(a) else float(np.mean(a > 60.)))
            rows[key] = row
    result = dict(kind="native_replay_loaded_wheel_camber_diagnostic", definition="asin(abs(worldZ dot wheel_localY_axle));0deg vertical wheel,90deg horizontal wheel",
        rows=rows, core_interval_s=[1.44,4.], load_threshold_strict_n=5.,
        descriptive_camber_threshold_deg=60., new_acceptance_gate=False,
        reference_target="Original reference wheel axles are horizontal; current tracking reward constrains world axial heading but not axle tilt.",
        inference="A wheel can carry load on a highly tilted cylinder. Loaded-wheel support and small root tilt do not establish upright tire-like rolling/sliding. The current heading-only reward permits this geometry.",
        task_success_claimed=False, new_physics_or_policy_steps=0,
        analyzer_sha256=sha(__file__), replay_states_sha256=sha(states_path), model_sha256=sha(args.model))
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--result",type=Path,required=True);p.add_argument("--model",type=Path,required=True)
    run(p.parse_args())
