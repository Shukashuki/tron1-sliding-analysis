"""Offline complete SA29 PPO/SAC physical friction-grid publication.

Inputs are a portable, explicit, hash-bound private capture tree, not a live
simulator. Checkpoint bytes are hashed, never deserialized. FK recovers only
wheel geometry; contact, COM velocities, actions and clocks remain measured.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from analyze_controller_pair import (audit_reports, audit_training, load_arrays,
    plant_source_provenance, posture_diagnostic, read_json, reference_trace,
    sha256, source_model, summarize_pair)
from analyze_nmpc_friction_development import case_name
from analyze_nmpc_nominal_development import require, safe_configuration, contact_diagnostic
from analyze_nmpc_sa29_pilot import CORE, EVENTS, FRAMES, SAMPLES, REFERENCE_ID
from analyze_policy_friction_dynamics import reference_numeric_representation_audit, audit_xy_replay
from analyze_tvlqr_pilot import PUBLIC_TRACE_FIELDS, PUBLIC_SIGNALS, public_compact
from controller_metrics import heading_signals
from controller_references import ROOT, DEFAULT_CATALOG, get_reference, verify_reference_pair
from controller_trace import WheelKinematics, native_trace
from nmpc_tracking_objective import JOINT_NAMES


GRID = (.01,.03,.05,.1,.15,.2,.3,.45,.6,.7,.8,1.,1.2,1.4,1.6,1.8,2.)
SELECTED_MU = (.01,.1,.7,1.6,2.)
TRANSITIONS = 12_288_000
FILE_ROLES = {"isaac_trace", "isaac_report", "isaac_policy_trace", "isaac_actor", "isaac_contract", "mujoco_trace", "mujoco_report"}
BASE_SOURCE_SHA256 = {
    "scripts/eval_tracking.py": "fd90089810d3d14bfd96210723afdd3a36113a19f667ec60b2df617dd032f992",
    "scripts/eval_tracking_mujoco.py": "da6d1da378b5065589a49f2befaebfb0576665dbd2ad5d34a157d11b8984a3aa",
    "scripts/train_tracking.py": "ea9e4cd14f25ff3ba5202b9ba43ac0f08320418e2311cbae205c8449f3a1da9c",
    "training/tron1_tracking.py": "0a29fa27c1d65c0b9add0886221491e549553e3529d6970f1ec5ba35e2532817",
    "training/tron1_drift.py": "d5da59c7af58f8a45e3deeb8f69fb0af8d382335e7b0974a5637106fc8c56bb6",
    "training/tron1_contact_slip.py": "52ef38de9cece19a8ee02c120cb85d472ffd831e4587e5f0d1f340974ab699f4",
    "training/tron1_domain_randomization.py": "3f3a0b62c85a0af496ae9e9e0d75dda85c3ec30da587e4a47969843f5b1b1dca",
    "training/tron1_physics_clock.py": "bcd89d6a24652d4501baf6744b46944d369fd224a9ab4727fc8b6073a20cf44e",
    "training/tron1_runtime_device.py": "2a1460d1b48e884fc5608e0365edc43b815d1fc74c8439945d4e2cc18027f775",
}
ADAPTER_SHA256 = {
    "train_beyondmimic_style.py": "880e84b6394ab99e51aa610ee9db778e113b173012a13db6e76284167983ca5d",
    "train_sac_tracking_isaac.py": "c485b155fbffc44ba7e8147bc76358ca5978e0b8aace81cf8c1d4b495c9d83cf",
    "sac_tracking.py": "694c0f280ff5fb89f74afa6a27992725dfb0725130e6575df8d3357b01c6b233",
    "terminal_observation.py": "fd67d968dc46ce96d01ccec8eda196b2e1e535092ad64d035c27eb210eb2a42a",
    "height_smoothness.py": "e133aca58f671fc22ae1dfc97a03461e7724e4072bf8b74f58a2c14281d198b8",
}
ANALYSIS_SOURCES=("analyze_sa29_policy_friction_dynamics.py","analyze_controller_pair.py","analyze_policy_friction_dynamics.py",
    "analyze_tvlqr_pilot.py","controller_trace.py","controller_metrics.py","controller_references.py","nmpc_tracking_objective.py",
    "materialize_controller_source.py","materialize_full_horizon_source.py","materialize_wide_friction_source.py","materialize_friction_evaluator.py")


def _digest(value):
    require(isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value), "Invalid SHA-256 identity")
    return value


def _relative(value):
    require(isinstance(value,str) and value and not Path(value).is_absolute()
            and not value.startswith(("~","file:")) and "\\" not in value and ":" not in value
            and all(part not in ("", ".", "..") for part in value.split("/")), "Require a portable relative capture path")
    return value


def checked_file(binding, base_dir):
    require(isinstance(binding,dict) and set(binding)=={"path","sha256"}, "File binding requires exactly path and sha256")
    base = Path(base_dir).resolve(); path = (base/_relative(binding["path"])).resolve()
    require(path.is_relative_to(base) and path.is_file() and path.stat().st_size>0, "Missing file or capture-root escape")
    require(sha256(path)==_digest(binding["sha256"]), "Captured file SHA-256 differs: "+Path(binding["path"]).name)
    return path


def parse_manifest(document, base_dir):
    require(isinstance(document,dict) and set(document)=={
        "schema_version","kind","reference_id","reference_spec","reference_catalog_sha256","evaluation_mu","inputs","evaluation_source","policies","cases"},
        "Unexpected or missing capture-manifest fields")
    spec=get_reference(REFERENCE_ID)
    require(document["schema_version"]==1 and document["kind"]=="sa29_policy_full_physical_grid_capture"
            and document["reference_id"]==REFERENCE_ID and document["reference_spec"]==spec
            and document["reference_catalog_sha256"]==sha256(DEFAULT_CATALOG)
            and document["evaluation_mu"]==list(GRID), "Only the complete locked SA29 17-coefficient grid is supported")
    require(set(document["inputs"])=={"motion","robot_reference","model"}, "Require three explicit locked physical/reference inputs")
    result={"inputs":{name:checked_file(value,base_dir) for name,value in document["inputs"].items()},"policies":{},"cases":[]}
    require(set(document["policies"])=={"PPO","SAC"}, "Require separately trained PPO and SAC")
    for algorithm, item in document["policies"].items():
        roles={"manifest","status","checkpoint","runner_config","policy_contract"}|({"actor"} if algorithm=="SAC" else set())
        require(isinstance(item,dict) and set(item)==roles, "Incomplete or unexpected training artifact bindings")
        paths={name:checked_file(value,base_dir) for name,value in item.items()}
        expected={"manifest":"manifest.json","status":"run_status.json","checkpoint":"model_final.pt","runner_config":"runner_config.json","policy_contract":"policy_contract.json","actor":"actor_normalized.pt"}
        require(len({p.parent for p in paths.values()})==1 and all(p.name==expected[n] for n,p in paths.items()),
                "Original adjacent training filenames must be preserved")
        result["policies"][algorithm]=paths
    source=document["evaluation_source"]
    require(isinstance(source,dict) and set(source)=={"root","audit","source_sha256"}, "Incomplete isolated evaluation-source binding")
    root=(Path(base_dir).resolve()/_relative(source["root"])).resolve()
    require(root.is_relative_to(Path(base_dir).resolve()) and root.is_dir(), "Evaluation source escapes the private capture tree")
    result["source_root"]=root
    require(isinstance(document["cases"],list) and len(document["cases"])==34, "Complete publication requires all 34 policy pairs")
    seen=set(); inodes=set()
    for item in document["cases"]:
        require(isinstance(item,dict) and set(item)=={"algorithm","mu","files","execution"}, "Unexpected case fields")
        algorithm,mu=item["algorithm"],item["mu"]
        require(algorithm in ("PPO","SAC") and type(mu) in (int,float) and mu in GRID
                and (algorithm,mu) not in seen and set(item["files"])==FILE_ROLES
                and set(item["execution"])=={"isaac","mujoco"}, "Duplicate/undeclared/incomplete policy case")
        files={name:checked_file(value,base_dir) for name,value in item["files"].items()}
        for engine, entry in item["execution"].items():
            require(isinstance(entry,dict) and set(entry)=={"receipt","log"}, "Require real per-engine terminal receipt and full log")
            for value in entry.values():
                path=checked_file(value,base_dir); state=path.stat(); key=(state.st_dev,state.st_ino)
                require(key not in inodes, "Execution evidence reused across distinct case roles"); inodes.add(key)
        for path in files.values():
            state=path.stat(); key=(state.st_dev,state.st_ino)
            require(key not in inodes, "The same physical file/inode is reused across distinct episode roles"); inodes.add(key)
        seen.add((algorithm,mu)); result["cases"].append({"algorithm":algorithm,"mu":float(mu),"files":files,"execution":item["execution"]})
    require(seen=={(a,m) for a in ("PPO","SAC") for m in GRID}, "Missing physical policy pairs")
    result["cases"].sort(key=lambda item:(item["algorithm"],item["mu"]))
    return result


def expected_evaluation_sources():
    """Compute the already-audited materializer outputs in memory, never write."""
    import materialize_wide_friction_source as wide
    import materialize_full_horizon_source as full
    import materialize_controller_source as controller
    original=ROOT/"upstream/tron1-learning-from-human"
    for name,value in BASE_SOURCE_SHA256.items():
        require(sha256(original/name)==value, "Packaged original TRON1 source changed: "+name)
    drift,evaluator=wide.extend_sources(original)
    changed={"training/tron1_drift.py":drift,
        "scripts/eval_tracking_friction_isaac.py":controller.patch_controller_entry(full.patch_isaac(evaluator)),
        "scripts/eval_tracking_mujoco.py":full.patch_mujoco((original/"scripts/eval_tracking_mujoco.py").read_text()),
        "training/tron1_full_horizon_diagnostic.py":(ROOT/"scripts/full_horizon_diagnostic.py").read_text(),
        "training/tron1_scripted_actor_runner.py":(ROOT/"scripts/scripted_actor_runner.py").read_text()}
    hashes={name:hashlib.sha256(text.encode()).hexdigest() for name,text in changed.items()}
    return {**BASE_SOURCE_SHA256,**hashes},hashes


def audit_source(root, audit_binding, source_sha256, base_dir):
    audit_path=checked_file(audit_binding,base_dir); audit=read_json(audit_path)
    expected,changed=expected_evaluation_sources()
    require(source_sha256==expected and audit.get("source_hashes")==BASE_SOURCE_SHA256
            and audit.get("changed_output_sha256")==changed and audit.get("evaluation_mu_bounds")==[.01,2.]
            and audit.get("full_horizon_diagnostic") is True and audit.get("training_unchanged") is True
            and audit.get("controller_algorithms")==["PPO","SAC"]
            and audit.get("materializer_sha256")==sha256(ROOT/"scripts/materialize_controller_source.py"),
            "Isolated source differs from the approved packaged d5da original and audited full-horizon/controller materializers")
    for name,value in expected.items():
        require(sha256(Path(root)/name)==value, "Changed captured evaluation source: "+name)
    return {"audit_sha256":sha256(audit_path),"original_source_sha256":dict(BASE_SOURCE_SHA256),
        "generated_source_sha256":expected,"changed_output_sha256":changed,"full_horizon_diagnostic":True,
        "source_difference_disclosure":"SA29 packaged original drift d5da59c7..., not historical primary formal drift 5eb40c9b..."}


def _device(values):
    require(isinstance(values,dict) and values.get("status")=="verified"
            and values.get("required_unmasked") is True and values.get("cuda_visible_devices")==None
            and values.get("native_and_tensor_device_consistency_verified") is True, "Missing verified unmasked native/tensor device audit")
    requested=values.get("requested_device"); require(isinstance(requested,str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)",requested), "Explicit CUDA ordinal required")
    index=int(requested.split(":")[1])
    require(all(values.get(k)==requested for k in ("environment_device","physics_sim_view_frontend_device","robot_native_transform_tensor_device"))
            and all(type(values.get(k)) is int and values[k]==index for k in ("native_physics_backend_device_ordinal","torch_current_device","physics_cudaDevice_setting"))
            and type(values.get("physics_sceneMultiGPUMode_setting")) is int and values["physics_sceneMultiGPUMode_setting"]==0,
            "CUDA native/tensor/settings ordinals differ")
    return safe_configuration(values)


def expected_ppo_runner(save_interval=100):
    """Read only the pinned literal config AST; never import the trainer."""
    path=ROOT/"upstream/tron1-learning-from-human/scripts/train_tracking.py"
    require(sha256(path)==BASE_SOURCE_SHA256["scripts/train_tracking.py"], "PPO config source changed")
    require(type(save_interval) is int and save_interval>0, "Invalid save-only interval")
    function=next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=="ppo_config")
    value=next(node.value for node in function.body if isinstance(node,ast.Return))
    class Parameters(ast.NodeTransformer):
        def visit_Name(self,node):
            require(node.id in ("seed","save_interval"), "PPO config must remain a literal with two declared parameters")
            return ast.Constant(42 if node.id=="seed" else save_interval)
    return ast.literal_eval(Parameters().visit(value))


def audit_training_budget(algorithm, training, status, runner, contract, checkpoint_sha256, actor_sha256=None):
    spec=get_reference(REFERENCE_ID)
    require(algorithm in ("PPO","SAC") and training.get("algorithm")==algorithm
            and training.get("reference_id")==REFERENCE_ID and training.get("reference_spec")
                =={**spec,"verification_scope":"motion_byte_identity_only_not_reference_pair_or_asset_validation"}
            and training.get("motion_sha256")==spec["motion_npz_sha256"]
            and training.get("reference_catalog_sha256")==sha256(DEFAULT_CATALOG)
            and training.get("pooled_success_statistics") is False, "Require separately trained SA29 policy, not primary checkpoint transfer")
    require(status.get("status")=="training_completed" and status.get("trained_policy_claimed") is True
            and status.get("smoke_only") is training.get("smoke_only") is False,
            "Require completed non-smoke training")
    require(type(training.get("seed")) is int and training["seed"]==42
            and type(training.get("num_envs")) is int and training["num_envs"]==512
            and type(status.get("environment_transitions")) is int and status["environment_transitions"]==TRANSITIONS
            and type(status.get("policy_steps_completed")) is int and status["policy_steps_completed"]==24000
            and training.get("environment_transitions_requested")==TRANSITIONS
            and training.get("policy_steps_requested")==24000, "Actual SA29 training budget/seed/environment count differs")
    require(contract==training.get("contract")==training.get("policy_contract")
            and contract.get("actor_dim")==51 and contract.get("action_dim")==8
            and training.get("height_adapter_sha256")==ADAPTER_SHA256["height_smoothness.py"], "Training contract/height adapter differs")
    profile_path=ROOT/"config/car_drift_height_smooth_task.json"; profile=read_json(profile_path)
    height=training.get("height_regularization",{})
    require(training.get("task_profile")==profile and training.get("task_profile_sha256")==sha256(profile_path)
            and height.get("mode")=="vertical_acceleration_only" and height.get("absolute_height_reward") is False
            and height.get("height_termination") is False and height.get("reference_vertical_velocity_reward") is False
            and all(height.get(k)==profile["height_regularization"][k] for k in ("mode","weight","acceleration_scale_m_s2"))
            and contract.get("height_regularization")==height, "Training task/reward/height-smoothing contract differs")
    expected_names={"training/tron1_tracking.py","training/tron1_drift.py","training/tron1_domain_randomization.py","training/tron1_contact_slip.py"}
    if algorithm=="PPO": expected_names.add("scripts/train_tracking.py")
    require(training.get("source_training_modules_sha256")=={n:BASE_SOURCE_SHA256[n] for n in expected_names}, "SA29 training source pins differ from packaged originals")
    if algorithm=="PPO":
        require(training.get("iterations")==1000 and training.get("num_steps_per_env")==24
                and training.get("setup_smoke_control_steps")==0 and status.get("iterations_requested")==1000
                and status.get("learning_iteration_index")==999 and runner.get("num_steps_per_env")==24
                and runner.get("seed")==42 and training.get("trainer_sha256")==ADAPTER_SHA256["train_beyondmimic_style.py"],
                "PPO actual rollout/iteration recipe differs")
        require(runner==expected_ppo_runner(runner.get("save_interval",100)), "PPO learner/normalization recipe differs")
    else:
        config={"actor_dim":51,"critic_dim":120,"action_dim":8,"hidden_sizes":[256,128,128],"activation":"elu",
            "learning_rate":.0003,"alpha_learning_rate":.0003,"initial_alpha":.004,"target_entropy":None,
            "tau":.005,"log_std_min":-5.,"log_std_max":2.,"max_grad_norm":10.}
        require(runner=={"algorithm":"SAC","sac":training.get("sac_config")}
                and training.get("sac_config")==config
                and all(training.get(k)==v for k,v in {"batch_size":1024,"gamma":.99,"warmup_transitions":10000,
                    "updates_per_policy_step":1,"replay_capacity":500000}.items())
                and training.get("comparison_modules_sha256")==ADAPTER_SHA256
                and status.get("checkpoint_sha256")==_digest(checkpoint_sha256)
                and status.get("actor_sha256")==_digest(actor_sha256)
                and type(status.get("learner_updates")) is int and status["learner_updates"]==23981,
                "SAC source/runner/checkpoint/actor/completed learner evidence differs")
    recipe=training.get("recipe",{})
    expected={"upstream_commit":"cd65172032893724b445448818c34165846d847d","friction_static":[.3,1.6],"friction_dynamic":[.3,1.2],
        "restitution":[0.,.5],"material_buckets":64,"joint_default_position_offset_rad":[-.01,.01],
        "base_com_offset_m":{"x":[-.025,.025],"y":[-.05,.05],"z":[-.05,.05]},"push_interval_s":[1.,3.],
        "push_velocity_range":{"x":[-.5,.5],"y":[-.5,.5],"z":[-.2,.2],"roll":[-.52,.52],"pitch":[-.52,.52],"yaw":[-.78,.78]},"ground_friction":1.}
    require(all(recipe.get(k)==v for k,v in expected.items()), "BeyondMimic DR recipe differs")
    return {"algorithm":algorithm,"seed":42,"num_envs":512,"actual_environment_transitions":TRANSITIONS,
        "actual_policy_steps":24000,"checkpoint_sha256":_digest(checkpoint_sha256),"actor_sha256":actor_sha256,
        "recipe":safe_configuration(recipe),"training_device_audit":_device(training.get("device_audit")),
        "source_training_modules_sha256":training["source_training_modules_sha256"],"smoke_only":False}


def _finite(raw,key,shape):
    value=np.asarray(raw[key]); require(value.shape==shape and value.dtype.kind in "biuf" and np.isfinite(value).all(), "Invalid complete physical field: "+key)
    return value


def audit_policy_trace(raw, engine, report):
    require(engine in ("isaac","mujoco"), "Unknown policy engine")
    time=_finite(raw,"time_s",(SAMPLES,))
    require(np.asarray(raw["physics_dt_s"]).shape==() and float(raw["physics_dt_s"])==.005
            and np.allclose(time,np.arange(SAMPLES)*.005,atol=1e-10,rtol=0), "Require original 1033 consecutive physical samples, 0..5.16 s")
    phase=np.r_[0,np.repeat(np.arange(FRAMES),4)]
    if engine=="isaac":
        for name,expected in (("valid_mask",np.ones((SAMPLES,1),bool)),("terminal_sample",np.r_[np.zeros(SAMPLES-1,bool),True][:,None]),
            ("action_was_applied",np.r_[False,np.ones(SAMPLES-1,bool)][:,None])):
            value=np.asarray(raw[name]); require(value.dtype.kind=="b" and np.array_equal(value,expected), "No partial/reset-spliced Isaac mask or terminal/action sample substitution")
        require(np.array_equal(_finite(raw,"simulation_step_counter",(SAMPLES,)),np.arange(SAMPLES))
                and np.array_equal(_finite(raw,"reference_frame_for_action",(SAMPLES,1))[:,0],phase)
                and np.array_equal(_finite(raw,"reference_frame",(SAMPLES,1))[:,0],phase), "Isaac counter/held phase differs")
        clipped=_finite(raw,"actions_clipped",(SAMPLES,1,8))[:,0]; actionraw=_finite(raw,"actions_raw",(SAMPLES,1,8))[:,0]
        require(np.array_equal(clipped,np.clip(actionraw,-1.,1.)) and np.array_equal(clipped[0],np.zeros(8)), "Isaac clipped/initial actions differ")
        decisions=clipped[1::4]
        require(np.array_equal(clipped[1:],np.repeat(decisions,4,axis=0))
                and np.array_equal(actionraw[1:],np.repeat(actionraw[1::4],4,axis=0)), "Isaac action is not held over exactly four substeps")
        require(report.get("reference_frames")==FRAMES and report.get("seed")==42
                and report.get("physics_recording",{}).get("samples")==SAMPLES, "Isaac report full clock differs")
        episodes=report.get("summary",{}).get("episodes",[])
        require(len(episodes)==1 and episodes[0].get("completed_full_reference") is True
                and episodes[0].get("recorded_samples")==FRAMES+1
                and np.isclose(episodes[0].get("recorded_duration_s",np.nan),5.16,atol=1e-9,rtol=0), "Isaac episode is not the complete SA29 reference")
    else:
        require(np.array_equal(_finite(raw,"reference_frame",(SAMPLES,)),phase)
                and np.array_equal(_finite(raw,"reference_frame_for_action",(FRAMES,)),np.arange(FRAMES))
                and np.allclose(_finite(raw,"policy_time_s",(FRAMES,)),time[::4][:-1],atol=1e-10,rtol=0)
                and float(raw["policy_dt_s"])==.02, "MuJoCo decision/held physical phases differ")
        clipped=_finite(raw,"actions_clipped",(FRAMES,8)); actionraw=_finite(raw,"actions_raw",(FRAMES,8))
        require(np.array_equal(clipped,np.clip(actionraw,-1.,1.))
                and np.array_equal(_finite(raw,"previous_clipped_actions",(FRAMES,8)),np.vstack((np.zeros((1,8)),clipped[:-1]))), "MuJoCo action clipping/memory differs")
        require(report.get("reference_frames")==FRAMES and report.get("policy_steps")==FRAMES
                and report.get("physics_steps")==SAMPLES-1 and report.get("completed_full_reference") is True
                and np.isclose(report.get("recorded_duration_s",np.nan),5.16,atol=1e-9,rtol=0), "MuJoCo full episode report differs")
    trace=native_trace(raw,"policy_"+engine)
    require(len(trace["time_s"])==SAMPLES, "Never truncate a longer/reset trace to a valid prefix")
    counts=trace["contact_count"]
    require(np.all(counts>=0) and np.array_equal(counts,np.round(counts)) and np.all(trace["contact_slip"]>=0), "Invalid recorded contact counts/slip")
    for name in ("nonwheel_ground_contact_count","nonwheel_ground_contact_force_norm_n"):
        value=_finite(raw,name,(SAMPLES,1) if engine=="isaac" else (SAMPLES,))
        value=value[:,0] if engine=="isaac" else value
        require(np.all(value>=0) and (not name.endswith("count") or np.array_equal(value,np.round(value))), "Invalid measured nonwheel ground evidence")
        trace[name]=value.copy()
    qpos=_finite(raw,"qpos",(SAMPLES,1,15) if engine=="isaac" else (SAMPLES,15))
    qpos=qpos[:,0] if engine=="isaac" else qpos
    require(np.allclose(qpos[:,:3],trace["position"],atol=1e-7,rtol=0)
            and np.allclose(np.abs(np.sum(qpos[:,3:7]*trace["quaternion"],axis=1)),1.,atol=1e-5,rtol=0), "Physical qpos and measured root pose differ")
    names=tuple(np.asarray(raw["joint_names"] if engine=="isaac" else raw["qpos_joint_names"]).tolist())
    require(len(names)==8 and set(names)==set(JOINT_NAMES)
            and np.allclose(qpos[:,7+np.array([names.index(n) for n in JOINT_NAMES])],trace["joint_pos"],atol=1e-7,rtol=0),
            "Physical qpos and named measured joints differ")
    return trace


def _argument(command,flag):
    require(command.count(flag)==1, "Missing/ambiguous actual execution flag: "+flag)
    index=command.index(flag); require(index+1<len(command), "Missing execution flag value")
    return command[index+1]


def audit_execution(execution, algorithm, mu, engine, files, base_dir):
    require(isinstance(execution,dict) and set(execution)=={"receipt","log"}, "Require actual per-engine execution receipt/log bindings")
    receipt_path=checked_file(execution["receipt"],base_dir); log_path=checked_file(execution["log"],base_dir)
    receipt=read_json(receipt_path); roles={r for r in FILE_ROLES if r.startswith(engine+"_")}
    require(receipt.get("schema_version")==1 and receipt.get("kind")=="sa29_policy_engine_terminal_receipt"
            and receipt.get("algorithm")==algorithm and receipt.get("mu")==mu and receipt.get("reference_id")==REFERENCE_ID
            and receipt.get("engine")==engine and receipt.get("terminal_status")=="exited"
            and type(receipt.get("exit_code")) is int and receipt["exit_code"]==0
            and receipt.get("log_sha256")==sha256(log_path)
            and receipt.get("artifact_sha256")=={r:sha256(files[r]) for r in roles}, "Terminal receipt identity/status/log/artifact bindings differ")
    log=log_path.read_text(errors="strict")
    require(not re.search(r"Traceback \(most recent call last\)|Fatal Python error|Segmentation fault|CUDA (?:error|out of memory)|unhandled exception|\[(?:ERROR|FATAL)\]",log,re.I),
            "Execution log contains an error/traceback despite terminal status")
    command=receipt.get("command")
    require(isinstance(command,list) and len(command)>=2 and all(isinstance(x,str) and x for x in command), "Missing actual execution argv")
    entry="eval_tracking_friction_isaac.py" if engine=="isaac" else "eval_tracking_mujoco.py"
    operand=1
    while operand<len(command) and command[operand] in ("-u","-B"): operand+=1
    require(re.fullmatch(r"python(?:3(?:\.[0-9]+)?)?(?:\.exe)?",Path(command[0]).name) is not None
            and operand<len(command) and Path(command[operand]).name==entry
            and sum(Path(x).name==entry for x in command)==1
            and not any(x in command for x in ("--allow-smoke-policy","--smoke-only","--max-policy-steps","--domain-parameters","--physics-dt-s","--deployment-leg-kp","--deployment-leg-kd")), "Wrong entrypoint or unsupported execution override")
    report=read_json(files[engine+"_report"])
    expected={"--motion-file":report.get("motion_file")}
    if engine=="isaac":
        expected.update({"--checkpoint":report.get("checkpoint"),"--runner-config":report.get("runner_config"),"--asset-path":report.get("asset_path")})
        require(_argument(command,"--num-envs")=="1" and _argument(command,"--seed")=="42"
                and float(_argument(command,"--evaluation-contact-friction"))==mu
                and _argument(command,"--device")==receipt.get("requested_device")
                ==report.get("runtime_device_audit",{}).get("requested_device")
                and all(command.count(flag)==1 for flag in ("--record-physics","--verify-unmasked-device","--headless")), "Isaac command contradicts captured fixed evaluation")
    else:
        expected.update({"--policy":report.get("policy"),"--contract":report.get("contract_source"),"--model":report.get("model")})
        require(receipt.get("requested_device")=="cpu" and command.count("--no-visual-mesh")==1
                and Path(_argument(command,"--verify-isaac-trajectory")).name=="trajectory.npz", "MuJoCo execution/Isaac-observation verification declaration differs")
    require(all(isinstance(value,str) and _argument(command,flag)==value for flag,value in expected.items()), "Recorded command contradicts actual report input paths")
    return {"receipt_sha256":sha256(receipt_path),"log_sha256":sha256(log_path),"exit_code":0,"terminal_status":"exited",
        "traceback_error_scan_passed":True,"scope":"Hash-bound original supervisor receipt and full log; not independent proof of provenance or GPU exclusivity"}


def audit_case_sources(ia, mj, sources, files, training):
    expected={"isaac":{"evaluator_source_sha256":"scripts/eval_tracking_friction_isaac.py","drift_module_sha256":"training/tron1_drift.py",
        "contact_slip_helper_sha256":"training/tron1_contact_slip.py","runtime_device_module_sha256":"training/tron1_runtime_device.py"},
        "mujoco":{"evaluator_source_sha256":"scripts/eval_tracking_mujoco.py","task_source_sha256":"training/tron1_tracking.py",
        "domain_source_sha256":"training/tron1_domain_randomization.py","contact_slip_helper_sha256":"training/tron1_contact_slip.py"}}
    for engine,report in (("isaac",ia),("mujoco",mj)):
        require(all(report.get(field)==sources[relative] for field,relative in expected[engine].items()), "Reported evaluator/helper source differs from captured source")
        for field,value in (("reference_id",REFERENCE_ID),("role","separate_supplement"),("stratum","high_sideslip_s_avoidance")):
            require(field not in report or report[field]==value, "Conflicting optional report reference identity")
    contract=read_json(files["isaac_contract"])
    require(contract==ia.get("policy_export")==mj.get("contract")
            and mj.get("contract_source_sha256")==sha256(files["isaac_contract"])
            and mj.get("policy_sha256")==sha256(files["isaac_actor"])
            and ia.get("asset_root_file_sha256")==training.get("asset_sha256"), "Actual exported actor/contract/asset identities differ")
    observation=mj.get("isaac_observation_validation",{})
    require(observation.get("status")=="passed" and observation.get("trajectory_sha256")==sha256(files["isaac_policy_trace"])
            and observation.get("observations_checked")==FRAMES+1 and observation.get("atol")==2e-5 and observation.get("rtol")==2e-5,
            "MuJoCo did not verify the captured complete Isaac policy-rate observation trajectory")
    export=contract.get("export_validation_max_abs_error")
    require(type(export) in (int,float) and np.isfinite(export) and 0<=export<=1e-4, "Actual export check differs from source evaluator limit")
    return {"reported_evaluator_source_sha256":{e:r["evaluator_source_sha256"] for e,r in (("isaac",ia),("mujoco",mj))},
        "exported_actor_sha256":sha256(files["isaac_actor"]),"exported_contract_sha256":sha256(files["isaac_contract"]),
        "isaac_policy_trajectory_sha256":sha256(files["isaac_policy_trace"]),"observation_replay_status":"passed",
        "isaac_device_audit":_device(ia.get("runtime_device_audit"))}


def audit_policy_trajectory(policy_raw, physical_raw):
    time=_finite(policy_raw,"time_s",(FRAMES+1,))
    require(np.allclose(time,np.arange(FRAMES+1)*.02,atol=1e-10,rtol=0)
            and float(policy_raw["policy_dt_s"])==.02, "Isaac policy-rate trajectory clock differs")
    for key in ("actions_raw","actions_clipped","action_observation","reference_frame_for_action","action_was_applied"):
        require(np.array_equal(policy_raw[key],physical_raw[key][::4]), "Policy-rate/physical sampled action evidence differs: "+key)
    require(np.asarray(policy_raw["valid_mask"]).dtype.kind=="b" and np.asarray(policy_raw["valid_mask"]).shape==(FRAMES+1,1)
            and np.asarray(policy_raw["valid_mask"]).all(), "Policy-rate trajectory is incomplete or reset-spliced")


def compact_case(data, prefix, require_nonwheel=False):
    require(type(require_nonwheel) is bool and (prefix=="reference" or re.fullmatch(r"(?:PPO|SAC)_mu[0-9]p[0-9]{3}_(?:isaac|mujoco)",prefix)), "Unknown public policy prefix")
    fields={name:np.asarray(data[prefix+"_"+name]) for name in PUBLIC_TRACE_FIELDS if prefix+"_"+name in data}
    needed={"time_s","physics_dt_s","joint_names","position","quaternion","com_velocity","angular_velocity","wheel_pos"}
    if prefix!="reference": needed|={"contact_count","contact_force","contact_slip"}
    if require_nonwheel: needed|={"nonwheel_ground_contact_count","nonwheel_ground_contact_force_norm_n"}
    require(needed<=set(fields), "Public projection lacks required measured fields")
    n,dt=(FRAMES,.02) if prefix=="reference" else (SAMPLES,.005)
    require(fields["time_s"].shape==(n,) and float(fields["physics_dt_s"])==dt
            and np.allclose(fields["time_s"],np.arange(n)*dt,atol=1e-10,rtol=0)
            and tuple(fields["joint_names"].tolist())==JOINT_NAMES, "Public SA29 native clock/named order differs")
    for name,shape in (("position",(n,3)),("quaternion",(n,4)),("com_velocity",(n,3)),("angular_velocity",(n,3)),("wheel_pos",(n,2,3))):
        require(fields[name].shape==shape and np.isfinite(fields[name]).all(), "Invalid public physical geometry")
    if prefix=="reference": require(not {"contact_count","contact_force","contact_slip"}&set(fields), "Reference contacts are unmeasured, not fabricated")
    else:
        require(fields["contact_count"].shape==(n,2) and fields["contact_force"].shape==(n,2,3) and fields["contact_slip"].shape==(n,2)
                and np.all(fields["contact_count"]>=0) and np.array_equal(fields["contact_count"],np.round(fields["contact_count"]))
                and np.all(fields["contact_slip"]>=0), "Invalid public measured contact projection")
    wheelq=np.asarray(data[prefix+"_wheel_quaternion_wxyz"]); wheelv=np.asarray(data[prefix+"_wheel_center_velocity_w"])
    signals=heading_signals(fields["quaternion"],fields["com_velocity"],wheelq,wheelv)
    require(all(np.array_equal(signals[k],data[prefix+"_"+k]) for k in PUBLIC_SIGNALS), "Public heading/speed/mask signals fail independent replay")
    for name in ("nonwheel_ground_contact_count","nonwheel_ground_contact_force_norm_n"):
        if name in fields:
            require(fields[name].shape==(n,) and np.isfinite(fields[name]).all() and np.all(fields[name]>=0)
                    and (not name.endswith("count") or np.array_equal(fields[name],np.round(fields[name]))), "Invalid public nonwheel contact evidence")
    return fields,{"signals":signals,"wheel_quaternion_wxyz":wheelq,"wheel_center_velocity_w":wheelv}


def _public_reference_proof(arrays, reference):
    motion={"body_names":np.array(["base_Link"]),"time_s":reference["time_s"],"body_pos_w":reference["position"][:,None]}
    robot={"time_s":arrays["reference_robot_time_s"],"root_target_position_m":arrays["reference_robot_position_float64_m"]}
    return reference_numeric_representation_audit(motion,robot)


FALSE_CLAIMS=("new_policy_evaluation","new_training","dynamics_stepping","learned_inference","inverse_kinematics","retargeting",
    "eligible_for_controller_ranking","controlled_execution_success_claimed","height_or_flight_failure_gate","heading_offset_is_mismatch",
    "pooled_success_statistics","hardware_ready","real_time_verified","qualified_freeze_created","robustness_certificate","translation_rotation_or_time_warp_applied")


def load_public_bundle(directory, spec, inputs, reference):
    directory=Path(directory); require({p.name for p in directory.iterdir()}=={"summary.json","policy_dynamics_trace.npz","topdown.png"}, "Public bundle must contain exactly three completed files")
    hashes={name:sha256(directory/name) for name in ("summary.json","policy_dynamics_trace.npz","topdown.png")}
    summary=read_json(directory/"summary.json"); arrays=load_arrays(directory/"policy_dynamics_trace.npz")
    require(summary.get("kind")=="complete_sa29_policy_physical_friction_grid" and summary.get("reference_spec")==spec==get_reference(REFERENCE_ID)
            and summary.get("input_sha256")==inputs and set(inputs)=={"motion","robot_reference","model"}
            and summary.get("evaluation_mu")==list(GRID) and summary.get("complete_predeclared_grid") is True
            and summary.get("paired_policy_cases")==34 and summary.get("physical_engine_episodes")==68
            and all(summary.get(name) is False for name in FALSE_CLAIMS), "Public policy grid identity/completion/scope differs")
    require(inputs["motion"]==spec["motion_npz_sha256"] and inputs["robot_reference"]==spec["robot_reference_npz_sha256"]
            and all(_digest(v) for v in inputs.values()) and summary.get("frames_per_episode")==FRAMES
            and summary.get("physical_samples_per_episode")==SAMPLES and summary.get("duration_s_per_episode")==5.16
            and summary.get("reference_catalog_sha256")==sha256(DEFAULT_CATALOG), "Public input/clock/catalog identities differ")
    for artifact in summary["artifacts"].values(): require(hashes[artifact["file"]]==artifact["sha256"], "Public artifact hash differs")
    require(set(summary["analysis_source_sha256"])==set(ANALYSIS_SOURCES), "Missing exact analysis-source role bindings")
    for name,value in summary["analysis_source_sha256"].items():
        require(Path(name).name==name and sha256(ROOT/"scripts"/name)==value, "Public analysis source changed")
    expected_sources,changed=expected_evaluation_sources(); source=summary["evaluation_source"]
    require(source["generated_source_sha256"]==expected_sources and source["original_source_sha256"]==BASE_SOURCE_SHA256
            and source["changed_output_sha256"]==changed and source["full_horizon_diagnostic"] is True
            and _digest(source["audit_sha256"]), "Public isolated-source bindings differ")
    prefixes=["reference"]+[a+"_"+case_name(mu)+"_"+engine for a in ("PPO","SAC") for mu in GRID for engine in ("isaac","mujoco")]
    allowed={"schema_version","wheel_order","reference_robot_time_s","reference_robot_position_float64_m"}|{
        p+"_"+field for p in prefixes for field in (*PUBLIC_TRACE_FIELDS,*PUBLIC_SIGNALS,"wheel_quaternion_wxyz","wheel_center_velocity_w")}
    require(set(arrays)<=allowed and all(not value.dtype.hasobject and (value.dtype.kind not in "biufc" or np.isfinite(value).all()) for value in arrays.values()), "Non-allowlisted/nonfinite public fields")
    ref,rg=compact_case(arrays,"reference")
    require(all(np.array_equal(ref[name],reference[name]) for name in ref if name in reference), "Public reference differs from locked source")
    require(_public_reference_proof(arrays,ref)==summary["reference_numeric_representation_audit"], "Public numeric reference proof differs")
    require(set(summary["runs"])=={a+"_"+case_name(mu) for a in ("PPO","SAC") for mu in GRID}, "Missing canonical policy pairs")
    for algorithm in ("PPO","SAC"):
        policy=summary["policies"][algorithm]; budget=policy["budget"]; training_hashes=policy["input_sha256"]
        require(set(training_hashes)==({"manifest","status","checkpoint","runner_config","policy_contract"}|({"actor"} if algorithm=="SAC" else set()))
                and all(_digest(v) for v in training_hashes.values()) and budget["checkpoint_sha256"]==training_hashes["checkpoint"], "Public training file/weight identities differ")
        require(budget["algorithm"]==algorithm and budget["actual_environment_transitions"]==TRANSITIONS
                and budget["actual_policy_steps"]==24000 and budget["seed"]==42 and budget["num_envs"]==512 and budget["smoke_only"] is False, "Public actual training evidence differs")
        for mu in GRID:
            label=algorithm+"_"+case_name(mu); run=summary["runs"][label]
            require(run["algorithm"]==algorithm and run["mu"]==mu and set(run["input_sha256"])==FILE_ROLES
                    and all(_digest(v) for v in run["input_sha256"].values()), "Public case capture identities differ")
            training=run["training"]; audit=run["audit"]; bindings=run["source_bindings"]
            require(training["manifest_sha256"]==training_hashes["manifest"] and training["run_status_sha256"]==training_hashes["status"]
                    and training["checkpoint_sha256"]==audit["checkpoint_sha256"]==budget["checkpoint_sha256"]
                    and training["checkpoint_bytes_verified"] is training["verified_formal_training"] is True
                    and training["algorithm"]==algorithm and audit["fixed_mu"]==mu and audit["eligible_for_controller_ranking"] is False,
                    "Public per-case training/checkpoint/coefficient identities contradict policy evidence")
            require(bindings["exported_actor_sha256"]==run["input_sha256"]["isaac_actor"]
                    and bindings["exported_contract_sha256"]==run["input_sha256"]["isaac_contract"]
                    and bindings["isaac_policy_trajectory_sha256"]==run["input_sha256"]["isaac_policy_trace"], "Public export/observation identities differ")
            for engine in ("isaac","mujoco"):
                execution=run["execution"][engine]
                require(type(execution["exit_code"]) is int and execution["exit_code"]==0 and execution["terminal_status"]=="exited"
                        and execution["traceback_error_scan_passed"] is True and _digest(execution["receipt_sha256"])
                        and _digest(execution["log_sha256"]) and audit["trace_bindings"][engine]["sha256"]==run["input_sha256"][engine+"_trace"],
                        "Public terminal/capture identity evidence contradicts the measured trace")
            pair={"reference":ref}; geometry={"reference":rg}
            for engine in ("isaac","mujoco"):
                pair[engine],geometry[engine]=compact_case(arrays,label+"_"+engine,require_nonwheel=True)
                require(posture_diagnostic(pair[engine])==run["posture_diagnostics"][engine], "Public posture diagnostic differs")
                require(contact_diagnostic(pair[engine],CORE)==run["contact_diagnostics"][engine]["core"], "Public measured contact diagnostic differs")
                for event in run["contact_diagnostics"][engine]["supplementary_events"]:
                    require(contact_diagnostic(pair[engine],event["interval_s"])==event["diagnostic"], "Public event contact differs")
            require(summarize_pair(spec,pair,geometry)==run["windows"], "Canonical SA29 core/events/gaps fail public replay")
    require(summary["rows"]==[{"mu":mu,**{a:summary["runs"][a+"_"+case_name(mu)]["windows"]["core"] for a in ("PPO","SAC")}} for mu in GRID], "Public all-17 table differs from canonical case windows")
    safe_configuration(summary)
    return summary,arrays,hashes


def render_topdown(compact, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(len(SELECTED_MU),3,figsize=(15,3.6*len(SELECTED_MU)),squeeze=False)
    for row,mu in enumerate(SELECTED_MU):
        bounds=[]
        for col,algorithm in enumerate(("reference","PPO","SAC")):
            ax=axes[row,col]
            for label,color in ([("reference",".35")] if col==0 else [(algorithm+"_"+case_name(mu)+"_isaac","#d62728"),(algorithm+"_"+case_name(mu)+"_mujoco","#1f77b4")]):
                t,pos,wheels=[compact[label+"_"+key] for key in ("time_s","position","wheel_pos")]
                core=(t>=CORE[0]-1e-9)&(t<=CORE[1]+1e-9)
                ax.plot(pos[:,0],pos[:,1],color=color,lw=.7,alpha=.3); ax.plot(pos[core,0],pos[core,1],color=color,lw=1.8)
                ids=np.arange(0,len(t),max(1,round(.4/float(compact[label+"_physics_dt_s"]))))
                for side in range(2):
                    for k in ids:
                        if not compact[label+"_root_course_to_wheel_valid"][k,side]:continue
                        heading=compact[label+"_wheel_rolling_plane_heading_rad"][k,side]; delta=.055*np.array([np.cos(heading),np.sin(heading)])
                        segment=wheels[k,side,:2]+np.array([-delta,delta])
                        eligible=label=="reference" or (compact[label+"_contact_count"][k,side]>0 and compact[label+"_contact_force"][k,side,2]>5.)
                        ax.plot(segment[:,0],segment[:,1],color=color,lw=.8,alpha=.9 if eligible else .12)
                ids=ids[compact[label+"_root_beta_valid"][ids]]; velocity=compact[label+"_com_velocity"][ids,:2]
                ax.quiver(pos[ids,0],pos[ids,1],velocity[:,0],velocity[:,1],color=color,angles="xy",scale_units="xy",scale=7,width=.003)
                bounds.extend((pos[:,:2],wheels[:,:,:2].reshape(-1,2)))
            ax.set(title=f"μ={mu:g} · "+("Reference (contacts UNMEASURED)" if col==0 else algorithm),xlabel="World X [m]",ylabel="World Y [m]")
            ax.set_aspect("equal",adjustable="box"); ax.grid(alpha=.2)
        xy=np.concatenate(bounds);lo,hi=xy.min(axis=0),xy.max(axis=0);margin=max(.15,.06*float(np.max(hi-lo)))
        for ax in axes[row]:ax.set_xlim(lo[0]-margin,hi[0]+margin);ax.set_ylim(lo[1]-margin,hi[1]+margin)
    fig.suptitle("SA29 policy recordings · separate S-avoidance / measured COM speed + contacts / not ranked",fontsize=11)
    fig.legend(handles=[Line2D([],[],color="#d62728",label="Isaac"),Line2D([],[],color="#1f77b4",label="MuJoCo"),
        Line2D([],[],color=".3",label="Thick: core; arrows: COM velocity; ticks: wheel planes; pale: no eligible contact")],loc="lower center",ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.975));fig.savefig(output,dpi=140);plt.close(fig)


def run(args):
    output=Path(args.output_dir); require(not output.exists(), "Preserve existing publication output")
    manifest_path=Path(args.manifest).resolve(); document=read_json(manifest_path); base=manifest_path.parent
    parsed=parse_manifest(document,base); inputs=parsed["inputs"]
    spec=verify_reference_pair(inputs["motion"],inputs["robot_reference"],REFERENCE_ID)
    source=document["evaluation_source"]; source_evidence=audit_source(parsed["source_root"],source["audit"],source["source_sha256"],base)
    motion=load_arrays(inputs["motion"]); robot=load_arrays(inputs["robot_reference"]); reference=reference_trace(motion)
    proof=reference_numeric_representation_audit(motion,robot); traces={"reference":reference}; policies={}; runs={}
    for algorithm,files in parsed["policies"].items():
        training,status,runner,contract=[read_json(files[name]) for name in ("manifest","status","runner_config","policy_contract")]
        budget=audit_training_budget(algorithm,training,status,runner,contract,sha256(files["checkpoint"]),sha256(files["actor"]) if "actor" in files else None)
        policies[algorithm]={"budget":budget,"input_sha256":{name:sha256(path) for name,path in files.items()}}
    with source_model(parsed["source_root"],inputs["model"]) as (model,loader):
        fk=WheelKinematics(model); recon={"reference":fk.reconstruct(reference)}
        for item in parsed["cases"]:
            algorithm,mu,files=item["algorithm"],item["mu"],item["files"]; label=algorithm+"_"+case_name(mu)
            ia,mj=read_json(files["isaac_report"]),read_json(files["mujoco_report"])
            training_files=parsed["policies"][algorithm]; training=read_json(training_files["manifest"])
            execution={e:audit_execution(item["execution"][e],algorithm,mu,e,files,base) for e in ("isaac","mujoco")}
            hashes={name:sha256(path) for name,path in files.items()}
            audit=audit_reports(algorithm,spec,ia,mj,isaac_trace_sha256=hashes["isaac_trace"],mujoco_trace_sha256=hashes["mujoco_trace"],
                model_sha256=sha256(inputs["model"]),mujoco_capture_sha256=hashes["mujoco_trace"])
            require(audit["fixed_mu"]==mu and ia.get("runner_config_sha256")==sha256(training_files["runner_config"]), "Fixed coefficient or trained runner identity differs")
            provenance=audit_training(training_files["manifest"],algorithm,spec,ia,mj,checkpoint_path=training_files["checkpoint"])
            source_bindings=audit_case_sources(ia,mj,source["source_sha256"],files,training)
            pair={"reference":reference}; geometry={"reference":recon["reference"]}; rawia=None
            for engine,report in (("isaac",ia),("mujoco",mj)):
                raw=load_arrays(files[engine+"_trace"]); trace=audit_policy_trace(raw,engine,report)
                if engine=="isaac":rawia=raw
                full=label+"_"+engine; traces[full]=pair[engine]=trace; recon[full]=geometry[engine]=fk.reconstruct(trace)
            audit_policy_trajectory(load_arrays(files["isaac_policy_trace"]),rawia)
            windows=summarize_pair(spec,pair,geometry)
            audit["eligible_for_controller_ranking"]=False
            runs[label]={"algorithm":algorithm,"mu":mu,"input_sha256":hashes,"execution":execution,"audit":audit,
                "training":provenance,"source_bindings":source_bindings,"windows":windows,
                "posture_diagnostics":{e:posture_diagnostic(pair[e]) for e in ("isaac","mujoco")},
                "contact_diagnostics":{e:{"core":contact_diagnostic(pair[e],CORE),"supplementary_events":[
                    {"interval_s":interval,"descriptive_not_gate":True,"diagnostic":contact_diagnostic(pair[e],interval)} for interval in EVENTS]} for e in ("isaac","mujoco")},
                "reported_source_provenance":{e:plant_source_provenance(r) for e,r in (("isaac",ia),("mujoco",mj))}}
    compact=public_compact(traces,recon);compact["reference_robot_position_float64_m"]=robot["root_target_position_m"].copy();compact["reference_robot_time_s"]=robot["time_s"].copy()
    summary={"schema_version":1,"kind":"complete_sa29_policy_physical_friction_grid","status":"offline_recording_analysis_only",
        "reference_id":REFERENCE_ID,"reference_spec":spec,"reference_catalog_sha256":sha256(DEFAULT_CATALOG),
        "capture_manifest_sha256":sha256(manifest_path),"input_sha256":{name:sha256(path) for name,path in inputs.items()},
        "evaluation_mu":list(GRID),"paired_policy_cases":34,"physical_engine_episodes":68,"complete_predeclared_grid":True,
        "frames_per_episode":FRAMES,"physical_samples_per_episode":SAMPLES,"duration_s_per_episode":5.16,
        "policies":policies,"runs":runs,"rows":[{"mu":mu,**{a:runs[a+"_"+case_name(mu)]["windows"]["core"] for a in ("PPO","SAC")}} for mu in GRID],
        "reference_numeric_representation_audit":proof,"evaluation_source":source_evidence,"model_loader_audit":loader,
        "fk_audit":{name:value["audit"] for name,value in recon.items()},"events_are_descriptive_not_success_gates":True,
        "raw_physical_recordings_retained_as_inputs":True,"raw_state_actions_or_weights_in_public_projection":False,
        **{name:False for name in FALSE_CLAIMS},"selected_display_mu":list(SELECTED_MU),
        "analysis_source_sha256":{name:sha256(ROOT/"scripts"/name) for name in ANALYSIS_SOURCES},
        "limitations":["Complete means recorded duration, not successful upright tracking; the core alone uses the existing XY screen.",
            "One seed/episode per coefficient, not statistical success rates, a controller ranking or a universal friction threshold.",
            "SA29 is CARLA S-avoidance, separate from the primary large-angle-turn task; no pooled statistics.",
            "Reference contacts are unmeasured; eligible physical contacts require count>0 and upward net force>5N. Unavailable slip is null, not zero or proof of flight.",
            "COM velocities and contact-point slip are recorded, not finite-differenced or re-simulated. FK reconstructs geometry only.",
            "Native world axes/clocks retained; exact f64-to-f32 reference cast proof is representation-only, never registration or time warp.",
            "Terminal receipts/source/hash bindings establish consistency of trusted captures, not independent authenticity or GPU context exclusivity."]}
    safe_configuration(summary)
    return publish_bundle(summary,compact,output,reference)


def publish_bundle(summary, compact, output, reference):
    """Stage three completed files; publish only after strict offline replay."""
    output=Path(output);require(not output.exists(), "Preserve existing publication output")
    require(summary.get("complete_predeclared_grid") is True and summary.get("paired_policy_cases")==34
            and summary.get("physical_engine_episodes")==68, "Never publish a partial policy grid")
    safe_configuration(summary)
    output.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix=output.name+".staging-",dir=output.parent))
    np.savez_compressed(stage/"policy_dynamics_trace.npz",**compact);render_topdown(compact,summary,stage/"topdown.png")
    summary["artifacts"]={name:{"file":name,"sha256":sha256(stage/name)} for name in ("policy_dynamics_trace.npz","topdown.png")}
    (stage/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    load_public_bundle(stage,summary["reference_spec"],summary["input_sha256"],reference)
    require(not output.exists(), "Publication target appeared while analyzing; preserve both captures")
    stage.rename(output)
    return summary


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest",type=Path,required=True);result.add_argument("--output-dir",type=Path,required=True)
    return result


if __name__=="__main__":
    result=run(parser().parse_args());print(json.dumps({"status":result["status"],"paired_policy_cases":result["paired_policy_cases"]}))
