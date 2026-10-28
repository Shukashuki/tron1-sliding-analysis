"""Audit all declared episodes; separate seed-level learning from reset success.

No success decision is taken from training reward or early episode resets.
Reconstruct native contacts and actor inputs/actions before using frozen gates.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import (load, sha, audit_actor_observations,
    reconstruct_contacts, derive_signals, qualify_stop, imitation_metrics)
from analyze_hockey_ppo import audit_observations, contact_event_summary
from hockey_speed_study import wilson_interval


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "protocol", "command-dir", "training-dir", "suite", "actor", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    read = lambda f: json.loads(Path(f).read_text())
    cfg = read(a.protocol)
    manifest = read(a.bundle/"manifest.json")
    for row in manifest["files"]:
        path = (a.bundle/row["path"]).resolve()
        if not path.is_relative_to(a.bundle.resolve()) or sha(path) != row["sha256"]:
            raise ValueError("Frozen study payload changed")
    training = read(a.training_dir/"report.json")
    suite = read(a.suite/"report.json")
    if training["transitions"] != 12288000 or training["status"] != "training_completed_not_task_success":
        raise ValueError("Require full-budget training")
    if suite["status"] != "validation_suite_completed_requires_independent_qualification":
        raise ValueError("Incomplete validation suite")
    expected = [r["id"] for r in cfg["evaluation_cases"]]
    if [r["id"] for r in suite["cases"]] != expected or len(set(expected)) != len(expected):
        raise ValueError("Missing, duplicate or extra evaluation cases")
    fields = {"config_sha256": a.protocol, "motion_sha256": a.command_dir/"motion.npz",
              "source_sha256": a.bundle/"train_hockey_ppo.py", "task_source_sha256": a.bundle/"hockey_ppo_task.py"}
    for report in (training, suite):
        if any(report[key] != sha(path) for key, path in fields.items()):
            raise ValueError("Training/evaluation identity differs")
        if any(report[key] != cfg[key] for key in ("method", "seed", "speed_scale")):
            raise ValueError("Study cell identity differs")
        if cfg["method"] == "beyondmimic" and report["baseline_adapter_sha256"] != sha(a.bundle/"hockey_beyondmimic_task.py"):
            raise ValueError("Baseline adapter differs")
        if cfg.get("friction_boundary") is not None and report.get("friction_boundary_source_sha256") != sha(a.bundle/"hockey_friction_boundary.py"):
            raise ValueError("Friction-boundary implementation differs")
    if not training["actor_sha256"] == suite["actor_sha256"] == sha(a.actor):
        raise ValueError("Different actor")
    reference = load(a.command_dir/"motion.npz")
    command = read(a.command_dir/"report.json")
    if command["motion_sha256"] != sha(a.command_dir/"motion.npz"):
        raise ValueError("Different evaluation target")
    import torch
    torch.set_num_threads(2)
    actor = torch.jit.load(str(a.actor), map_location="cpu").eval()
    a.output.mkdir(parents=True)
    rows = []
    for case, entry in zip(cfg["evaluation_cases"], suite["cases"]):
        directory = a.suite/case["id"]
        run = read(directory/"report.json")
        if sha(directory/"report.json") != entry["report_sha256"]:
            raise ValueError("Case report changed")
        for key in fields:
            if run[key] != suite[key]:
                raise ValueError("Case source mismatch")
        if run["worlds"] != 1 or run["steps"] != 139 or run["external_forces"] != 0:
            raise ValueError("Incorrect physical validation")
        for name, digest in run["artifacts"].items():
            if sha(directory/name) != digest:
                raise ValueError("Case artifact changed")
        reset = read(directory/"case.json")
        if reset["case"] != case or reset["maximum_initial_state_error"] > 5e-5:
            raise ValueError("Reset trial mismatch")
        material = run["material_audit"]
        if not (material["live_physx_materials_verified"] and material["ground_material_usd_verified"]
                and np.allclose(material["wheel_mu_range"], [.7, .7], atol=1e-6, rtol=0)
                and all(g["friction"] == 1 and g["combine"] == "multiply" for g in material["ground_colliders"])):
            raise ValueError("Different evaluation friction")
        trace, raw = load(directory/"final_rollout.npz"), load(directory/"native_contact_buffers.npz")
        if trace["root_pos"].shape != (1113, 3) or trace["actions"].shape != (139, 8):
            raise ValueError("Incomplete physical horizon")
        obs_audit = (audit_observations(trace, reference, cfg) if cfg["method"] == "hockey"
                     else audit_actor_observations(trace, reference))
        with torch.inference_mode():
            replay = actor(torch.as_tensor(trace["observations"], dtype=torch.float32)).clamp(-1, 1).numpy()
        action_error = float(abs(replay-trace["actions"]).max())
        if not np.isfinite(action_error) or action_error > 5e-5:
            raise ValueError("Recorded action does not match trained actor")
        contact, native_audit = reconstruct_contacts(trace, raw, .7)
        signals = derive_signals(trace, contact)
        qualification = qualify_stop(trace, signals, reference, command)
        imitation, _ = imitation_metrics(trace, signals, reference, command)
        first_fall, events = contact_event_summary(signals, command)
        result = dict(case=case, input_sha256=run["artifacts"], run_report_sha256=sha(directory/"report.json"),
            qualification=qualification, imitation=imitation, first_45deg_tilt_s=first_fall,
            observation_audit=obs_audit, action_replay_maximum_error=action_error,
            native_reconstruction=native_audit, contact_events=events)
        (a.output/(case["id"]+".json")).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        np.savez_compressed(a.output/(case["id"]+".signals.npz"), **signals)
        rows.append(dict(id=case["id"], kind=case["kind"], success=qualification["task_success_claimed"],
            failed_gates=qualification["failed_gates"], xy_rmse_m=qualification["xy_RMSE_m"],
            xy_peak_m=qualification["xy_peak_m"], first_45deg_tilt_s=first_fall,
            report_sha256=sha(a.output/(case["id"]+".json"))))
    perturbed = [r for r in rows if r["kind"] == "perturbed"]
    n, k = len(perturbed), sum(r["success"] for r in perturbed)
    failures = {}
    for row in perturbed:
        for gate in row["failed_gates"]:
            failures[gate] = failures.get(gate, 0)+1
    report = dict(kind="hockey_speed_success_cell", method=cfg["method"], seed=cfg["seed"],
        speed_scale=cfg["speed_scale"], formal_transitions=training["transitions"],
        nominal_success=rows[0]["success"], perturbation_successes=k, perturbation_trials=n,
        perturbation_success_rate=k/n, conditional_Wilson_95_interval=wilson_interval(k, n),
        failure_gate_counts_nonexclusive=failures, cases=rows,
        success_definition="all eleven frozen physical stop-and-slide gates", nominal_in_random_denominator=False,
        confidence_scope="conditional on this trained policy and specified IID initial-state distribution; not training-seed reliability",
        training_report_sha256=sha(a.training_dir/"report.json"), suite_report_sha256=sha(a.suite/"report.json"),
        evaluator_sha256=sha(Path(__file__).with_name("evaluate_hockey_physx.py")),
        analyzer_sha256=sha(__file__), source_manifest_sha256=sha(a.bundle/"manifest.json"),
        pointwise_friction_cone_certified=False)
    if cfg.get("friction_boundary") is not None:
        report["friction_boundary"] = cfg["friction_boundary"]
    (a.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("method", "seed", "speed_scale", "nominal_success", "perturbation_success_rate")}))


if __name__ == "__main__":
    main()
