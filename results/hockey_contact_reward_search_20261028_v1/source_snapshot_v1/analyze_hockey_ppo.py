"""Qualify directly trained hockey PPO with the unchanged physical stop gates.

Run locally with CPU Torch. Independently reconstruct actor inputs/actions and
native contact metrics before applying the same stop/slide/posture evaluator.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import (sha, load, q_matrix, audit_actor_observations,
    reconstruct_contacts, derive_signals, qualify_stop, imitation_metrics, render_signals, segments)


def contact_event_summary(signals, command):
    """Separate physical slip from high-beta slip, and exclude post-fall events.

    These are diagnostics; the frozen qualification gates remain unchanged.
    Force samples represent completed substep intervals, not isolated poses.
    """
    s = signals
    t = s["time_s"]
    dt = float(np.median(np.diff(t)))
    fall = np.flatnonzero(s["root_tilt_deg"] >= 45.)
    first_fall = float(t[fall[0]]) if len(fall) else None
    valid = s["native_sample_valid"]
    pre_fall = np.ones(len(t), dtype=bool) if first_fall is None else t < first_fall
    braking = ((t-dt >= command["brake_start_s"]-1e-9)
               & (t <= command["brake_end_s"]+1e-9))
    rows = []
    for k, side in enumerate(("left", "right")):
        loaded = valid & (s["normal_fz"][:, k] > 5.)
        beta = loaded & s["wheel_beta_valid"][:, k]
        window = loaded & braking & pre_fall
        slip = window & (s["normal_point_lateral_rms"][:, k] > .05)
        saturated = slip & (s["aggregate_utilization"][:, k] >= .9) & (s["aggregate_utilization"][:, k] <= 1.05)
        saturated &= (s["anchor_power"][:, k] < -.05) & (s["dissipative_slipping_anchor_count"][:, k] >= 1)
        high_beta = saturated & s["wheel_beta_valid"][:, k] & (abs(s["wheel_axial_beta_deg"][:, k]) >= 20.)
        def events(mask):
            values = segments(t, mask, interval_samples=True)
            return dict(events=values, total_duration_s=float(mask.sum()*dt),
                        longest_duration_s=max((v["duration_s"] for v in values), default=0.))
        def quantiles(mask):
            return np.quantile(abs(s["wheel_axial_beta_deg"][mask, k]), [.5, .95, 1.]).tolist() if mask.any() else None
        rows.append(dict(side=side, loaded_beta_abs_deg_p50_p95_max=quantiles(beta),
            pre_fall_loaded_beta_abs_deg_p50_p95_max=quantiles(beta & pre_fall),
            pre_fall_braking_lateral_slip=events(slip),
            pre_fall_braking_saturated_dissipative_slip=events(saturated),
            pre_fall_braking_high_beta_saturated_dissipative_slip=events(high_beta)))
    return first_fall, rows


def audit_observations(trace, reference, protocol):
    base = dict(trace, observations=trace["observations"][:, :51])
    report = audit_actor_observations(base, reference)
    n = len(trace["actions"])
    q = trace["root_quat"][np.arange(n)*8]
    w, x, y, z = q.T
    angle = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    v = reference["body_lin_vel_w"][:n, 0]
    # Replay the archived command's actual numerical semantics. V1 used
    # float32 time: frame 20 can be .39999998, delaying the braking flag by
    # one frame. V2 uses integer frame thresholds and fixes that boundary.
    if protocol.get("phase_clock") == "integer_reference_frames":
        index = np.arange(n)
        start, end = [round(protocol[k]/.02) for k in ("brake_start_s", "brake_end_s")]
        braking, holding = (index >= start)&(index < end), index >= end
    else:
        t = np.arange(n, dtype=np.float32)*np.float32(.02)
        braking = (t >= np.float32(protocol["brake_start_s"]))&(t < np.float32(protocol["brake_end_s"]))
        holding = t >= np.float32(protocol["brake_end_s"])
    target = np.column_stack((np.cos(angle)*v[:, 0]+np.sin(angle)*v[:, 1],
        -np.sin(angle)*v[:, 0]+np.cos(angle)*v[:, 1], reference["body_ang_vel_w"][:n, 0, 2],
        braking, holding))
    if trace["observations"].shape != (n, 56):
        raise ValueError("Wrong actor input ABI")
    error = abs(trace["observations"][:, 51:]-target)
    if not np.isfinite(error).all() or error.max() > 5e-5:
        raise ValueError("Explicit hockey command does not match source/physical state")
    return dict(base=report, extra_field_maximum_error=float(error.max()), all_56_fields_reconstructed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "training-dir", "actor", "source", "command-dir", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    run = json.loads((a.run_dir/"report.json").read_text())
    training = json.loads((a.training_dir/"report.json").read_text())
    protocol = json.loads((a.source/"protocol.json").read_text())
    manifest = json.loads((a.source/"manifest.json").read_text())
    for row in manifest["files"]:
        path = (a.source/row["path"]).resolve()
        if not path.is_relative_to(a.source.resolve()) or sha(path) != row["sha256"]:
            raise ValueError("Training source snapshot changed")
    for report in (run, training):
        for key, name in (("source_sha256", "train_hockey_ppo.py"), ("task_source_sha256", "hockey_ppo_task.py"),
                          ("config_sha256", "protocol.json"), ("motion_sha256", "motion.npz")):
            if report[key] != sha(a.source/name):
                raise ValueError("Run/source identity mismatch: "+key)
    if not (run["mode"] == "eval" and run["single_world_validation"] and run["worlds"] == 1
            and run["external_forces"] == 0 and run["steps"] == 139 and training["mode"] == "formal"
            and training["status"] == "training_completed_not_task_success"
            and run["variant"] == training["variant"]
            and run["actor_sha256"] == training["actor_sha256"] == sha(a.actor)):
        raise ValueError("Missing directly trained single-world validation identity")
    files = ("final_rollout.npz", "native_contact_buffers.npz")
    for name in files:
        if sha(a.run_dir/name) != run["artifacts"][name]:
            raise ValueError("Recording changed")
    material = run["material_audit"]
    if not (material["live_physx_materials_verified"] and material["ground_material_usd_verified"]
            and np.allclose(material["wheel_mu_range"], [.7, .7], atol=1e-6, rtol=0)
            and all(g["friction"] == 1 and g["combine"] == "multiply" for g in material["ground_colliders"])):
        raise ValueError("Unexpected nominal material")
    reference = load(a.command_dir/"motion.npz")
    command = json.loads((a.command_dir/"report.json").read_text())
    if sha(a.command_dir/"motion.npz") != run["motion_sha256"] or command["motion_sha256"] != run["motion_sha256"]:
        raise ValueError("Wrong hockey reference")
    trace, raw = [load(a.run_dir/n) for n in files]
    if len(trace["root_pos"]) != 1113 or len(trace["actions"]) != 139:
        raise ValueError("Incomplete full-horizon trace")
    observation_audit = audit_observations(trace, reference, protocol)
    import torch
    actor = torch.jit.load(str(a.actor), map_location="cpu").eval()
    with torch.inference_mode():
        predicted = actor(torch.tensor(trace["observations"], dtype=torch.float32)).clamp(-1, 1).numpy()
    action_error = float(abs(predicted-trace["actions"]).max())
    if not np.isfinite(action_error) or action_error > 5e-5:
        raise ValueError("Recorded action does not match trained actor")
    contact, native_audit = reconstruct_contacts(trace, raw, .7)
    signals = derive_signals(trace, contact)
    qualification = qualify_stop(trace, signals, reference, command)
    imitation, _ = imitation_metrics(trace, signals, reference, command)
    first_fall, contact_events = contact_event_summary(signals, command)
    report = dict(kind="direct_hockey_ppo_qualification_v1", variant=run["variant"],
        run_report_sha256=sha(a.run_dir/"report.json"), training_report_sha256=sha(a.training_dir/"report.json"),
        analyzer_sha256=sha(__file__), physical_evaluator_sha256=sha(Path(__file__).with_name("evaluate_hockey_physx.py")),
        input_sha256={n: sha(a.run_dir/n) for n in files}, actor_sha256=sha(a.actor),
        actor_observation_audit=observation_audit, replay_maximum_action_error=action_error,
        native_reconstruction=native_audit, qualification=qualification, imitation=imitation,
        first_45deg_tilt_diagnostic_s=first_fall, loaded_beta_summary=contact_events,
        mu=.7, physics_rerun=False, pointwise_cone_saturation_certified=False,
        friction_indicator="wheel aggregate ratio with same-anchor dissipative sliding corroboration")
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/"signals.npz", **signals)
    render_signals(signals, a.output/"contact_beta_4x1.png", "direct hockey PPO / "+run["variant"])
    (a.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(qualification, indent=2))


if __name__ == "__main__":
    main()
