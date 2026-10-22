"""Recorded-pose four-panel PPO versus CEM with matched 400 Hz stage reward.

Both plants are native CPU MuJoCo at mu=.7. PPO was trained in PhysX; CEM
uses a privileged native dynamics model. There is no new policy inference,
physics stepping, pose correction, or manufactured PPO episode return here.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np

import render_ppo_beta_physx_to_mujoco_400hz as paired
from controller_metrics import heading_signals
from controller_trace import native_trace
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES, HeightSmoothTrackingObjective
from ppo_backend_400hz_contract import read_json, require, sha256, write_json
from render_physx_ppo_fullbody import build_qpos, create_runtime, mesh_provenance, _pose
from render_ppo_backend_400hz_nominal import diagnostics, replay_fk_audit
from render_ppo_beta_nominal import world_bounds
from render_ppo_fixed_follow import prepare_camera, apply_camera
from render_ppo_training_recipe_comparison import _map_panel
from summarize_ppo_backend_400hz_nominal import local_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CEM = ROOT/"outputs/ppo_cem_matched_reward_400hz_20261022_v1"
ORDER = ("ppo_beta_on", "cem_beta_on", "ppo_beta_off", "cem_beta_off")
TITLES = {"ppo_beta_on": "PPO | beta ON (2) | PhysX-trained",
    "cem_beta_on": "MPC-CEM | beta ON (2) | native prediction",
    "ppo_beta_off": "PPO | beta OFF (0) | PhysX-trained",
    "cem_beta_off": "MPC-CEM | beta OFF (0) | native prediction"}
SHORT = {"ppo_beta_on": "PPO ON", "cem_beta_on": "CEM ON", "ppo_beta_off": "PPO OFF", "cem_beta_off": "CEM OFF"}
COLORS = {"ppo_beta_on": "#2365a4", "cem_beta_on": "#dc7927", "ppo_beta_off": "#418d78", "cem_beta_off": "#995ea1"}
CANVAS_SIZE = (2560, 750)
native_indices = paired.native_indices


def read_decisions(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, list) and all(isinstance(row, dict) for row in value), "Require list-valued CEM decision log")
    return value


def check_clock_and_actions(raw):
    require(np.asarray(raw["time_s"]).shape == (2793,)
        and float(raw["physics_dt_s"]) == .0025 and float(raw["policy_dt_s"]) == .02
        and np.allclose(raw["time_s"], np.arange(2793)*.0025, atol=1e-10, rtol=0),
        "Require complete native400Hz clock, without resetting or resampling")
    actions = np.asarray(raw["actions"])
    require(actions.shape == (349, 8) and np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1.+1e-9,
        "Require349 bounded native CEM actions")
    require(np.array_equal(raw["reference_frame_for_action"], np.r_[0, np.repeat(np.arange(349), 8)])
        and np.array_equal(raw["held_action"], np.vstack((np.zeros((1,8)), np.repeat(actions, 8, axis=0)))),
        "CEM held-action/reference phase differs from eight native substeps")
    history = np.asarray(raw["contact_force_history_w"])
    require(history.shape == (2793,3,9,3) and np.isfinite(history).all()
        and np.array_equal(history[0], np.zeros((3,9,3)))
        and np.array_equal(history[1:,-1], raw["body_net_forces_w"][1:])
        and np.array_equal(history[:-1,1:], history[1:,:-1]),
        "CEM must preserve measured native three-substep contact history")


def recompute_cem_reward(raw, report, decisions, reference_file):
    """Complete CEM logs permit exact stage rewards; PPO logs do not."""
    require(len(decisions) == 349, "Incomplete CEM solver/reward log")
    objective = HeightSmoothTrackingObjective(report["task_profile"], dt=.02,
        soft_joint_limits=report["soft_joint_pos_limits"])
    require(objective.contract() == report["objective_contract"], "CEM objective mathematical contract differs")
    with np.load(reference_file, allow_pickle=False) as z:
        names = tuple(z["body_names"].tolist())
        ids = [names.index(name) for name in BODY_NAMES]
        full_reference = {name: z[name][:,ids].copy() for name in
            ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    def state(index):
        result = {name: np.asarray(raw[name][index]) for name in
            ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "body_net_forces_w", "contact_force_history_w")}
        result.update(body_names=BODY_NAMES, joint_pos=raw["joint_pos"][index])
        return result
    previous = np.zeros(8)
    previous_vz = float(raw["body_lin_vel_w"][0,0,2])
    stage_cost, stage_reward = [], []
    for frame, decision in enumerate(decisions):
        require(decision["frame"] == frame and np.array_equal(decision["action"], raw["actions"][frame]),
            "CEM decisions/action log differs")
        pre, post = state(frame*8), state((frame+1)*8)
        reference = {name: value[frame] for name,value in full_reference.items()}
        held = objective.hold_reference(reference, pre)
        actual = objective.evaluate(post, held, raw["actions"][frame], previous, previous_vz, reset_sample=frame == 0)
        # The original scalar bridge explicitly discloses surrogate soft limits;
        # supplying its actual limits here suppresses only that disclosure flag.
        expected = copy.deepcopy(decision["objective"])
        expected.pop("surrogate_flags")
        observed = copy.deepcopy(actual); observed.pop("surrogate_flags")
        require(observed == expected, "Recomputed CEM fifteen-term stage reward differs")
        stage_cost.append(actual["stage_cost"])
        stage_reward.append(.02*actual["total_reward"])
        previous = np.asarray(decision["action"])
        previous_vz = actual["base_com_vz"]
    require(np.array_equal(raw["stage_cost"], stage_cost) and np.array_equal(raw["stage_reward"], stage_reward)
        and report["actual_CEM_reward_sum"] == float(np.sum(stage_reward)), "CEM reward totals differ")
    return {"all_349_stages_and_15_terms_recomputed_exactly": True,
        "actual_dt_scaled_CEM_reward_sum": float(np.sum(stage_reward)),
        "PPO_episode_return_available": False}


def load_comparison(cem_root=DEFAULT_CEM, off_root=paired.DEFAULT_OFF, on_root=paired.DEFAULT_ON,
                    reference=paired.DEFAULT_REFERENCE, model=paired.DEFAULT_MODEL,
                    legacy_evaluator=paired.DEFAULT_LEGACY, protocol=paired.DEFAULT_PROTOCOL):
    ppo, reference_values, evidence = paired.load_comparison(off_root, on_root, reference, model, legacy_evaluator, protocol)
    cem_root = Path(cem_root).resolve()
    experiment_file = local_file(cem_root, "experiment.json")
    experiment_sha = sha256(experiment_file); experiment = read_json(experiment_file)
    require(experiment["schema_version"] == 1 and experiment["kind"] == "same_stage_reward_400hz_ppo_cem_nominal_comparison"
        and experiment["status"] == "completed" and experiment["new_PPO_training"] is False
        and experiment["stage_rewards_copied_from_completed_PPO"] is True
        and experiment["PPO_episode_returns_available"] is False
        and experiment["comparison_is_optimizer_only"] is False,
        "Require completed400Hz matched-stage-reward experiment, not smoke or relabeling")
    require(experiment["PPO_input_evidence"] == evidence, "CEM experiment is bound to different PPO evidence")
    source_ledger = experiment["source_sha256"]
    require(len(source_ledger) >= 10, "Missing CEM captured source/configuration ledger")
    for name,digest in source_ledger.items():
        require(sha256(local_file(ROOT, name)) == digest, "Captured CEM source/configuration changed: "+name)
    settings = experiment["settings"]
    require(settings["physics_dt_s"] == .0025 and settings["decimation"] == 8 and settings["policy_dt_s"] == .02
        and settings["plant_mu"] == .7 and settings["prediction_mu_scenarios"] == [.7]
        and settings["discount_factor"] == .99 and settings["action_rate_weight"] == -.5
        and settings["grounded_reward_weight"] == .4 and settings["beta_conditions"] == ["beta_on", "beta_off"],
        "CEM frozen nominal settings differ")
    config = settings["nmpc"]
    require(config["horizon_steps"] == 30 and config["control_blocks"] == 10 and config["candidates"] == 64
        and config["iterations"] == 3 and config["elite_count"] == 8 and config["seed"] == 42
        and config["initial_std"] == [.04]*6+[.35]*2
        and config["min_std"] == [.005]*6+[.03]*2
        and config["parameterization"] == "warm_start_residual_blocks", "CEM search configuration changed")
    traces, rows, ledger = {}, {}, {"experiment.json": experiment_sha}
    fk, fk_source = paired.metrics.load_fk(model, legacy_evaluator)
    ppo_rows = {row["id"]: row for row in evidence["rows"]}
    for label in ("beta_on", "beta_off"):
        ppo_row = copy.deepcopy(ppo_rows[label]); ppo_id = "ppo_"+label
        traces[ppo_id] = ppo[label]
        ppo_row.update(id=ppo_id, title=TITLES[ppo_id], controller="PPO", training_backend="PhysX",
            validation_backend="native CPU MuJoCo", exact_episode_reward_recomputed=False)
        rows[ppo_id] = ppo_row
        paths = {name: local_file(cem_root, str(Path(label)/name)) for name in ("report.json", "rollout.npz", "solver_log.json")}
        hashes = {name: sha256(path) for name,path in paths.items()}
        ledger.update({label+"/"+name: value for name,value in hashes.items()})
        report = read_json(paths["report.json"]); raw = paired.read_raw(paths["rollout.npz"])
        require(experiment["reports"][label] == hashes["report.json"] and report["rollout_sha256"] == hashes["rollout.npz"]
            and report["solver_log_sha256"] == hashes["solver_log.json"], "CEM raw/report/solver hashes differ")
        require(report["status"] == "completed" and report["condition"] == label
            and report["kind"] == "400hz_CEM_with_completed_PPO_stage_reward" and report["engine"] == "native CPU MuJoCo"
            and report["mujoco_version"] == "3.8.1"
            and report["completed_policy_steps"] == report["max_policy_steps"] == 349
            and report["native_samples"] == 2793 and np.isclose(report["duration_s"], 6.98, atol=1e-10, rtol=0)
            and report["truncated_smoke"] is False and report["N_per_condition"] == 1,
            "CEM physical episode is incomplete, truncated, or mislabeled")
        require(report["physics_dt_s"] == .0025 and report["policy_dt_s"] == .02 and report["decimation"] == 8
            and report["plant_mu"] == .7 and report["prediction_mu_scenarios"] == [.7]
            and report["discount_factor"] == .99 and report["CEM_config"] == config
            and report["model_sha256"] == sha256(model) and report["motion_sha256"] == sha256(reference)
            and report["matched_PPO_report_sha256"] == ppo_row["report_sha256"],
            "CEM clock/model/reference/search/PPO identity differs")
        for name in ("PPO_actions_or_future_states_used_for_initial_plan", "new_training_or_policy_inference", "new_GPU_job",
                     "termination_or_reset_after_initialization", "height_contact_tilt_failure_gate", "PPO_total_reward_recomputed",
                     "real_time_verified", "controller_tuned_on_this_episode"):
            require(report[name] is False, "Unexpected CEM provenance/gate: "+name)
        require(report["native_plant_arrays_match_PPO_validation_exactly"] is True and report["native_contact_substeps"] == 8,
            "CEM native plant equivalence was not checked")
        require(tuple(raw["body_names"].tolist()) == BODY_NAMES and tuple(raw["action_joint_names"].tolist()) == JOINT_NAMES,
            "CEM measured body/action array ordering differs")
        ppo_report = read_json(Path(ppo_row["directory"])/"report.json")
        require(report["task_profile"] == ppo_report["contract"]["task_profile"], "CEM stage-reward task profile differs from PPO")
        for name in ("plant_contact_material_audit", "prediction_contact_material_audit"):
            material = report[name]; expected = ppo_report["contact_material_audit"]
            for field in ("native_plant_readback", "geom_friction", "geom_priority", "geom_condim", "geom_solref", "geom_solimp", "pair_coefficient"):
                require(material[field] == expected[field], "CEM/PPO compiled native plant/material differs: "+field)
        ppo_raw = paired.read_raw(Path(ppo_row["directory"])/"rollout.npz")
        require(np.array_equal(raw["qpos"][0], ppo_raw["initial_qpos"])
            and np.array_equal(raw["qvel"][0], ppo_raw["initial_qvel"])
            and np.array_equal(raw["qpos"][0], report["initial_state"]["qpos"])
            and np.array_equal(raw["qvel"][0], report["initial_state"]["qvel"]), "CEM did not preserve recorded PPO reset state")
        check_clock_and_actions(raw)
        rewards = recompute_cem_reward(raw, report, read_decisions(paths["solver_log.json"]), reference)
        trace = native_trace(raw, "nmpc"); reconstruction = fk.reconstruct(trace)
        trace.update(signals=reconstruction["signals"], wheel_quaternion_wxyz=reconstruction["wheel_quaternion_wxyz"],
            wheel_center_velocity_w=reconstruction["wheel_center_velocity_w"])
        trace["body_heading_geometry_valid"], trace["wheel_heading_geometry_valid"] = paired.metrics.heading_geometry_masks(trace)
        case_id = "cem_"+label; traces[case_id] = trace
        compute = np.asarray(raw["compute_s"])
        require(compute.shape == (349,) and np.isfinite(compute).all() and np.all(compute >= 0.)
            and np.array_equal(1000*np.quantile(compute,[.5,.95,1.]), report["compute_ms_median_p95_max"])
            and float(np.mean(compute > .02)) == report["compute_over_control_dt_fraction"], "CEM measured solve-time summary differs")
        rows[case_id] = {"id": case_id, "title": TITLES[case_id], "controller": "MPC-CEM", "training_backend": None,
            "validation_backend": "native CPU MuJoCo", "beta_reward_weight": 2. if label == "beta_on" else 0.,
            "directory": str(paths["report.json"].parent), "recording_sha256": hashes["rollout.npz"],
            "report_sha256": hashes["report.json"], "solver_log_sha256": hashes["solver_log.json"],
            "windows": paired.trace_metrics(trace, reference_values), "wheel_FK_audit": reconstruction["audit"],
            "exact_episode_reward_recomputed": True, "CEM_stage_reward_audit": rewards,
            "compute_ms_median_p95_max": report["compute_ms_median_p95_max"],
            "compute_over_control_dt_fraction": report["compute_over_control_dt_fraction"], "solver_failures": report["solver_failures"],
            "prediction_reset_caveat": report["prediction_reset_caveat"],
            "concurrent_local_CPU_episodes": report["concurrent_local_CPU_episodes"], "timing_caveat": report["timing_caveat"]}
    require(sha256(experiment_file) == experiment_sha, "CEM experiment changed during offline reading")
    for name,digest in ledger.items():
        require(sha256(local_file(cem_root,name)) == digest, "CEM inputs changed during offline reading")
    return {name: traces[name] for name in ORDER}, reference_values, {
        "schema_version": 1, "kind": "four_panel_400hz_matched_stage_reward_PPO_CEM_native_MuJoCo",
        "case_order": list(ORDER), "rows": [rows[name] for name in ORDER], "mu": .7,
        "validation_backend": "native CPU MuJoCo", "PPO_training_backend": "PhysX", "CEM_prediction_backend": "native CPU MuJoCo",
        "physics_dt_s": .0025, "policy_dt_s": .02, "decimation": 8, "native_samples_per_case": 2793,
        "policy_decisions_per_case": 349, "action_rate_weight": -.5, "original_grounded_reward_weight": .4,
        "PPO_input_evidence": evidence, "CEM_input_ledger_sha256": ledger, "CEM_source_sha256": source_ledger,
        "reference_sha256": sha256(reference), "model_sha256": sha256(model), "FK_provenance": fk_source,
        "stage_reward_function_and_weights_matched": True, "discount_factor": .99,
        "CEM_horizon_steps": 30, "CEM_terminal_value": None, "PPO_total_reward_recomputed": False,
        "comparison_is_optimizer_only": False, "same_native_plant_and_recorded_initial_state": True,
        "dense_ground_support_reward_used": False, "height_or_airborne_failure_gate": False,
        "full_horizon_preserved": True, "state_interpolation": False, "pose_alignment": False,
        "statistical_causal_effect_claimed": False, "sample_size_per_case": 1,
        "lower_HF_alone_establishes_stability": False, "new_training": False, "new_simulation": False,
        "new_policy_inference": False, "visual_reviewed": False,
        "limitations": ["Same mathematical15term stage reward and weights, not identical return optimization: CEM gamma=.99 for30steps(.6s), no terminalvalue; PPO has a learned discounted value.",
            "PPO was trained in PhysX with DR; CEM uses full simulator state/reference preview and a native MuJoCo predictor. This is not an optimizer-only comparison.",
            "Only CEM has full measured all-body contact history for exact episode stage-reward reconstruction. PPO total returns are unavailable, not imputed.",
            "Native400Hz, mu=.7, original reset state and compiled plant readbacks match. MuJoCo contact/soft-limit observables need not equal PhysX training observations.",
            "CEM predictors cold-reset solver warmstart/time every20ms; the evaluation plant integrates continuously. Same native engine is not identical hidden contact state.",
            "CEM wall time includes IPC and two concurrent local CPU episode jobs; it is not hardware-matched isolated PPO inference latency.",
            "One episode per condition: no statistical superiority, realtime or hardware claim. Core accuracy alone can hide later falls; all original poses remain visible.",
            "Persistent tilt is diagnostic only. Contact-count/Fz zeros are not flight labels; there is no absolute-height/contact failure gate.",
            "Reference ends6.96s: full6.98s tracking is unavailable without extrapolation; tilt/contact/HF retain the full native horizon."]}


def compact_arrays(traces, reference):
    result = {"reference_"+name: np.asarray(value).copy() for name,value in reference.items() if isinstance(value,np.ndarray)}
    for name in ORDER:
        result.update({name+"_"+field: np.asarray(traces[name][field]).copy() for field in paired.TRACE_FIELDS})
    return result


def recompute_compact(arrays):
    reference = {name[len("reference_"):]: value for name,value in arrays.items() if name.startswith("reference_")}
    result = {}
    for name in ORDER:
        trace = {field: np.asarray(arrays[name+"_"+field]) for field in paired.TRACE_FIELDS}
        trace["signals"] = heading_signals(trace["quaternion"], trace["com_velocity"], trace["wheel_quaternion_wxyz"], trace["wheel_center_velocity_w"])
        result[name] = paired.trace_metrics(trace, reference)
    return result


def draw_diagnostics(traces, reference, destination):
    from plot_ppo_cem_matched_reward_400hz import draw_diagnostics as draw
    return draw(traces, reference, destination)


def render_frame(runtime,traces,qposes,reference,summary,plan,frame):
    from PIL import Image, ImageDraw
    require(type(frame) is int and 0 <= frame < 350,"Invalid50Hz replay frame")
    index = int(native_indices()[frame]); image = Image.new("RGB",CANVAS_SIZE,"#fafafa"); draw = ImageDraw.Draw(image)
    draw.text((12,5),f"Matched stage reward | PPO vs MPC-CEM | native CPU MuJoCo | mu=0.7 |400Hz physics /50Hz control | t={frame*.02:.2f}s",font=runtime.font,fill="#111111")
    for column,name in enumerate(ORDER):
        camera=apply_camera(runtime,plan,name,frame)
        _pose(runtime.model,runtime.data,qposes[name][index])
        runtime.renderer.update_scene(runtime.data,camera=camera,scene_option=runtime.option)
        x=column*640; image.paste(Image.fromarray(runtime.renderer.render().copy()),(x,58))
        draw.rectangle((x,30,x+640,57),fill="#f0f2f4"); draw.text((x+9,34),TITLES[name],font=runtime.font,fill="#111111")
        value=diagnostics(traces[name],reference,index)
        xy=f"{value['xy_error_m']:.3f}m" if value["xy_error_m"] is not None else "n/a (ref ends6.96s)"
        draw.text((x+9,424),f"XY {xy} | speed {value['speed_m_s']:.2f}m/s | tilt {value['tilt_deg']:.1f}deg",font=runtime.font,fill="#111111")
        draw.text((x+9,447),f"Original height {value['height_m']:.3f}m | fixed camera, original pose",font=runtime.font,fill="#111111")
        draw.text((x+9,470),f"L/R count {int(value['count'][0])}/{int(value['count'][1])} | Fz {value['fz'][0]:.0f}/{value['fz'][1]:.0f}N",font=runtime.font,fill="#111111")
        display={key:value[native_indices()] if isinstance(value,np.ndarray) and value.ndim and len(value)==2793 else value for key,value in traces[name].items() if key!="signals"}
        display["signals"]={key:value[native_indices()] if isinstance(value,np.ndarray) and value.ndim and len(value)==2793 else value for key,value in traces[name]["signals"].items()}
        _map_panel(draw,display,reference,frame,(x+5,500,x+635,672),summary["world_bounds"],runtime.small)
    draw.text((12,683),"All panels: original grounded +0.4, action-rate -0.5, free absolute height. Body and L/R wheel headings are shown on identical world-XY maps.",font=runtime.small,fill="#333333")
    draw.text((12,703),"Same stage reward is not the same controller information or horizon: CEM uses full state/model, 0.6s preview, gamma=0.99; PPO uses its learned policy.",font=runtime.small,fill="#333333")
    draw.text((12,723),"N=1; retain full6.98s including falls. No pose edits/inference/dynamics during replay. CEM latency is offline; PPO episode returns unavailable.",font=runtime.small,fill="#333333")
    return image


def run(args):
    output=Path(args.output_dir); require(not output.exists(),"Preserve evidence; fresh output directory required")
    os.environ.setdefault("MUJOCO_GL","osmesa"); require(os.environ["MUJOCO_GL"]=="osmesa","Require CPU OSMesa graphics")
    traces,reference,summary=load_comparison(args.cem_root,args.off_root,args.on_root,args.reference,args.model,args.legacy_mujoco_evaluator,args.protocol)
    runtime=create_runtime(args.model,width=640,height=360)
    try:
        qposes,fk_audits={},{}
        for name in ORDER:
            qposes[name],ids=build_qpos(runtime.model,traces[name])
            fk_audits[name]={**replay_fk_audit(runtime.model,traces[name],qposes[name]),"named_joint_qpos_addresses":ids.tolist()}
        plan=prepare_camera(runtime,traces,qposes,ORDER,native_indices())
        summary.update(world_bounds=[value.tolist() for value in world_bounds(traces,reference)],
            fixed_follow_camera={key:value for key,value in plan.items() if key!="lookat_by_case"},
            mesh_provenance=mesh_provenance(args.model),fk_audits=fk_audits,renderer_sha256=sha256(__file__),
            video_frames=350,video_fps=50,video_size_pixels=list(CANVAS_SIZE),native_sample_stride=8,
            preview_original_time_s=5.,robot_mesh_alpha=1.)
        compact=compact_arrays(traces,reference); recomputed=recompute_compact(compact)
        require(all(recomputed[row["id"]]==row["windows"] for row in summary["rows"]),"Compact metrics must exactly reproduce all four windows")
        output.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(output/"native_comparison_signals.npz",**compact)
        replay={"fixed_camera_distance_m":np.array(plan["distance_m"])}
        for name in ORDER:
            replay.update({name+"_qpos":qposes[name],name+"_selected_native_indices":native_indices(),
                name+"_native_time_s":traces[name]["time_s"],name+"_camera_lookat_m":plan["lookat_by_case"][name]})
        np.savez_compressed(output/"replay_states.npz",**replay)
        render_frame(runtime,traces,qposes,reference,summary,plan,250).save(output/"comparison.png")
        render_frame(runtime,traces,qposes,reference,summary,plan,150).save(output/"comparison_core_3s.png")
        draw_diagnostics(traces,reference,output/"turn_diagnostics.png")
        rows=paired.overview_rows(summary)
        by_id={row["id"]:row for row in summary["rows"]}
        for row in rows:
            case=by_id[row["id"]]; row["controller"]=case["controller"]
            row["CEM_compute_ms_median"]=case.get("compute_ms_median_p95_max",[None])[0]
        with (output/"metric_table.csv").open("x",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
        write_json(output/"overview.json",{"kind":summary["kind"],"mu":.7,"sample_size_per_case":1,
            "stage_reward_function_and_weights_matched":True,"PPO_total_reward_recomputed":False,"visual_reviewed":False,"rows":rows})
        files=["native_comparison_signals.npz","replay_states.npz","comparison.png","comparison_core_3s.png","turn_diagnostics.png","metric_table.csv","overview.json"]
        if not args.skip_video:
            import imageio_ffmpeg
            stream=imageio_ffmpeg.write_frames(str(output/"ppo_cem_matched_reward_400hz.mp4"),CANVAS_SIZE,fps=50,
                codec="libx264",quality=7,macro_block_size=2,output_params=["-pix_fmt","yuv420p","-movflags","+faststart"])
            stream.send(None)
            try:
                for frame in range(350): stream.send(np.asarray(render_frame(runtime,traces,qposes,reference,summary,plan,frame)))
            finally: stream.close()
            files.append("ppo_cem_matched_reward_400hz.mp4")
        summary["video_generated"]=not args.skip_video
        summary["compact_native_recomputed_all_windows_exactly"]=True
        summary["helpers_sha256"]={name:sha256(Path(__file__).with_name(name)) for name in
            ("render_ppo_beta_physx_to_mujoco_400hz.py","render_ppo_fixed_follow.py","render_physx_ppo_fullbody.py",
             "render_ppo_training_recipe_comparison.py","analyze_ppo_backend_400hz_matrix.py","controller_trace.py","controller_metrics.py",
             "plot_ppo_cem_matched_reward_400hz.py")}
        summary["artifacts"]={name:sha256(output/name) for name in files}
        write_json(output/"summary.json",summary)
        return summary
    finally: runtime.renderer.close()


def parser():
    result=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    result.add_argument("--cem-root",type=Path,default=DEFAULT_CEM)
    result.add_argument("--off-root",type=Path,default=paired.DEFAULT_OFF)
    result.add_argument("--on-root",type=Path,default=paired.DEFAULT_ON)
    result.add_argument("--reference",type=Path,default=paired.DEFAULT_REFERENCE)
    result.add_argument("--model",type=Path,default=paired.DEFAULT_MODEL)
    result.add_argument("--legacy-mujoco-evaluator",type=Path,default=paired.DEFAULT_LEGACY)
    result.add_argument("--protocol",type=Path,default=paired.DEFAULT_PROTOCOL)
    result.add_argument("--output-dir",type=Path,required=True)
    result.add_argument("--skip-video",action="store_true")
    return result


if __name__=="__main__": run(parser().parse_args())
