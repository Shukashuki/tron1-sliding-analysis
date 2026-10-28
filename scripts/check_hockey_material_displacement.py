"""Cross-check pre-step slip with transport of the same wheel material points.

This uses the interval displacement already reconstructed from pre/post poses
by audit_hockey_contact_timing.py. A nonzero pre-solver velocity alone could
represent motion arrested by friction; material displacement checks whether
motion persists across the step. It does not certify uninterrupted substep
contact, and the mean contact-point vector is not a pointwise cone test.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, q_matrix, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    report_path = a.audit / "report.json"
    report = json.loads(report_path.read_text())
    signals_path = a.audit / "aligned_signals.npz"
    assert sha(signals_path) == report["aligned_signals_sha256"]
    archive = load(signals_path)
    result = dict(script_sha256=sha(__file__), audit_sha256=sha(report_path),
                  signals_sha256=sha(signals_path), physical_dt_s=.0025,
                  threshold_m_s=.05, window_s=[.4, 1.8], physics_steps=0,
                  scope="Mean material-point displacement across a step; contact certified at its start only. Not a new task success gate.",
                  cells=[])
    for cell in report["cells"]:
        name = cell["cell"]
        s = {k.split("__", 1)[1]: v for k, v in archive.items() if k.startswith(name + "__")}
        axis = q_matrix(s["wheel_quat_pre"])[..., :, 1].copy()
        axis[..., 2] = 0
        axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-8)
        lateral = np.abs((s["slip_velocity_interval"] * axis).sum(-1))
        row = dict(cell=name, wheels=[])
        for side in (0, 1):
            contact = s["braking"] & s["touching"][:, side]
            corroborated = contact & s["friction_corroborated_lateral_sliding"][:, side]
            moved = lateral[:, side] > .05
            row["wheels"].append(dict(side=("left", "right")[side],
                interval_lateral_motion_s=float((contact & moved).sum()*.0025),
                pre_slip_friction_and_interval_motion_s=float((corroborated & moved).sum()*.0025),
                interval_lateral_speed_p50_p95_max_m_s=np.quantile(lateral[contact, side], [.5, .95, 1]).tolist()))
        result["cells"].append(row)
    output = a.audit / "material_displacement_crosscheck.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
