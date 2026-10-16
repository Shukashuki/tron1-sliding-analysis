"""LOCAL offline planning-prefix diagnostic, not another plant/controller trial.

Reconstruct at most frames 0..137 of the original failed Isaac I6 episode,
feeding its recorded 111-D measured inputs into the UNMODIFIED archived CEM.
Planning integrates MuJoCo candidates; it never starts Isaac or advances a
validation plant. Stop at the first declared guard mismatch. No retry/restart,
retuning, new build or relaxed continuation is implemented.

--prepare-only copies/verifies source bytes without importing MuJoCo or doing
planning. Original full plans/elite score vectors were not saved: matching
available prefix evidence is NOT independent proof of historical plan bytes.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import tarfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINAL_FRAME = 137
MAX_CANDIDATE_EVALUATIONS = 53405
MAX_PREDICTED_SUBSTEPS = 6408600
ARCHIVE_SHA256 = "19a210cee30add02de5888645273320f3c983ef0d3d4ab4cf3a27039c0083535"
SNAPSHOT_MANIFEST_SHA256 = "332ac602375d5f1e50a32d7a5707949cceb8a77ddda7f873bc5a1cf74302dd2b"
NATIVE_SHA256 = "ce676e8414d38beb0676e8529cd88ebfea980ef9d91494c052e21414c4f89ea5"
NATIVE_MANIFEST_SHA256 = "7b60a686c358eb608a2aeae3b5b67a0a00513f9c4306c3ab69592f3499b84cc7"
EPISODE_SHA256 = {
    "report.json": "88c8cde0734687209ffb0487e055a16acd305a1c824e1de4a3e8dd43e3a4a01a",
    "rollout.npz": "5f7a251fe47e694a03e477a327298ea7bae02b520d0c7e531e8e59e869f764c4",
    "solver_log.json": "89dbb8b6408124d43c8a3cf426cd1b17d3215931a49bb48fe7f531b776eae120",
}
MEMBERS = (
    "scripts/run_nmpc_tracking_isaac.py", "scripts/run_nmpc_tracking.py", "scripts/nmpc_shooting.py",
    "scripts/nmpc_tracking_objective.py", "scripts/nmpc_batch_objective.py", "scripts/nmpc_native_batch.py",
    "scripts/classical_tracking.py", "scripts/controller_references.py", "scripts/height_smoothness.py",
    "config/controller_references.json", "config/nmpc_height_smooth.json",
    "config/nmpc_height_smooth_residual.json", "config/classical_tracking.json",
    "config/car_drift_height_smooth_task.json", "native/nmpc_batch.cpp",
)
HELPERS = ("scripts/eval_tracking_mujoco.py", "scripts/eval_tracking.py",
           "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py",
           "training/tron1_tracking.py", "training/tron1_drift.py",
           "training/tron1_runtime_device.py", "training/tron1_physics_clock.py")
COUNT_FIELDS = ("candidate_evaluations", "finite_candidates", "failed_candidates",
                "scenario_evaluations", "failed_scenarios", "batch_prediction_calls",
                "failed_batch_prediction_calls", "failed_scenarios_by_mu")
EXACT_FIELDS = ("initial_candidate_source", "warm_start_used", "optimization_failed", "fallback_used",
                "initial_candidate_generation_failed", "fallback_kind", "action_parameterization")
COST_FIELDS = ("initial_model_cost", "selected_model_cost", "selected_scenario_costs", "best_cost_by_iteration")


class ReconstructionStop(Exception):
    """Diagnostic-only stop; deliberately NOT caught by archived CEM."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def source_hash(report, relative):
    matches = [v for k, v in report["source_sha256"].items() if k.endswith("/"+relative)]
    if len(matches) != 1:
        raise ValueError("Missing/ambiguous original source role: "+relative)
    return matches[0]


def extract_snapshot(archive, destination, expected_hashes, *, archive_sha256=ARCHIVE_SHA256):
    """Mechanical exact-member copy; reject traversal, links and overwrites."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Preserve existing source snapshot: "+str(destination))
    if sha(archive) != archive_sha256 or set(expected_hashes) != set(MEMBERS):
        raise ValueError("Original archive/member contract differs")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) != 15 or {m.name for m in members} != set(MEMBERS):
            raise ValueError("Expected exactly the original fifteen source/config members")
        payload = {}
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe source member: "+member.name)
            data = bundle.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected_hashes[member.name]:
                raise ValueError("Original source-member hash differs: "+member.name)
            payload[member.name] = data
    destination.mkdir(parents=True)
    for relative, data in payload.items():
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    return dict(expected_hashes)


def finite(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Missing/malformed finite "+name)
    return result


def validate_recorded_inputs(trace, rows):
    states = finite(trace["packed_controller_states"], (349,111), "packed states")
    actions = finite(trace["actions"], (349,8), "actual actions")
    times = finite(trace["time_s"], (1397,), "physical clock")
    velocity = finite(trace["base_com_velocity_w"], (1397,3), "actual COM velocity")
    forces = finite(trace["body_net_forces_w"], (1397,9,3), "actual contact forces")
    if len(rows) != 349 or not np.allclose(times, np.arange(1397)*.005, atol=1e-12, rtol=0):
        raise ValueError("Incomplete original phase/physical clock")
    if not np.array_equal(np.diff(trace["simulation_step_counter"]), np.ones(1396)):
        raise ValueError("Original physical counter skipped/reset")
    if np.max(abs(actions)) > 1 or not np.array_equal(actions, actions.astype(np.float32).astype(float)):
        raise ValueError("Actual actions are not bounded exact float32 values")
    if not np.allclose(np.linalg.norm(states[:,3:7], axis=-1), 1., atol=1e-5, rtol=0):
        raise ValueError("Packed quaternions are not normalized")
    if not np.array_equal(states[:,-1], np.r_[1.,np.zeros(348)]):
        raise ValueError("Reset-memory flags differ")
    history = states[:,29:110].reshape(349,3,9,3)
    if not np.array_equal(history[:,0], forces[4*np.arange(349)]):
        raise ValueError("Packed newest-first contact memory differs")
    for k, row in enumerate(rows):
        if row["frame"] != k or row["reference_frame_for_reward"] != k:
            raise ValueError("Recorded reference phase differs")
        if not np.array_equal(row["action"], actions[k]):
            raise ValueError("Solver/trace actual actions differ")
        raw = finite(row["raw_cem_action"], (8,), "raw CEM action")
        if not np.array_equal(raw.astype(np.float32).astype(float), actions[k]):
            raise ValueError("Raw action did not produce recorded float32 action")
        if k and not np.array_equal(history[k], forces[4*k-np.arange(3)]):
            raise ValueError("Original contact history must remain newest first")
        vz = row["actual_isaac_objective"]["base_com_vz"]
        if not np.isfinite(vz) or float(vz) != float(velocity[4*(k+1),2]):
            raise ValueError("Logged actual previous COM-vz differs from direct physical trace")
    return states, actions


def previous_inputs(frame, trace, rows):
    if not isinstance(frame, (int,np.integer)) or isinstance(frame, bool) or not 0 <= frame <= FINAL_FRAME:
        raise ValueError("Only the fixed original prefix 0..137 is in scope")
    if frame == 0:
        return np.zeros(8), float(trace["base_com_velocity_w"][0,2])
    return np.asarray(trace["actions"][frame-1], dtype=float).copy(), float(rows[frame-1]["actual_isaac_objective"]["base_com_vz"])


def costs_equal(actual, expected, *, atol=0., rtol=0.):
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, (list,tuple)) or isinstance(expected, (list,tuple)):
        return (isinstance(actual, (list,tuple)) and isinstance(expected, (list,tuple))
                and len(actual) == len(expected)
                and all(costs_equal(a,b,atol=atol,rtol=rtol) for a,b in zip(actual,expected)))
    return bool(np.isfinite(actual) and np.isfinite(expected) and np.isclose(actual,expected,atol=atol,rtol=rtol))


def decision_mismatches(raw_action, info, expected, *, cost_atol=0., cost_rtol=0.):
    """Declared fixed guards; no tolerance adaptation or warm-plan overwrite."""
    mismatch = []
    if not np.array_equal(raw_action, expected["raw_cem_action"]):
        mismatch.append("raw_cem_action")
    if not np.array_equal(np.asarray(raw_action).astype(np.float32).astype(float), expected["action"]):
        mismatch.append("applied_float32_action")
    old = expected["solver"]
    for key in (*COUNT_FIELDS,*EXACT_FIELDS):
        if info.get(key) != old.get(key):
            mismatch.append(key)
    for key in COST_FIELDS:
        if not costs_equal(info.get(key), old.get(key), atol=cost_atol, rtol=cost_rtol):
            mismatch.append(key)
    return mismatch


def worker_status(worker):
    process = getattr(worker, "_process", None)
    result = {"closed": getattr(worker, "_closed", None), "alive": None, "pid": None, "exitcode": None}
    if process is not None:
        for key, read in (("alive", process.is_alive), ("pid", lambda: process.pid), ("exitcode", lambda: process.exitcode)):
            try:
                result[key] = read()
            except Exception:
                pass
    log = Path(worker._workdir.name)/"MUJOCO_LOG.TXT" if hasattr(worker,"_workdir") else None
    result["mujoco_log_tail"] = log.read_text(errors="replace")[-2000:] if log is not None and log.is_file() else ""
    result["scope"] = "Observed after exception, before outer cleanup; original internal pre-stop state was not recorded"
    return result


def replay_prefix(controller, observer, trace, rows, output, *, cost_atol=0., cost_rtol=0.):
    """Generic fail-closed loop, also testable without any simulator imports."""
    matched = []
    for frame in range(FINAL_FRAME+1):
        observer.begin(frame)
        previous,vz = previous_inputs(frame,trace,rows)
        rng_before = copy.deepcopy(controller.rng.bit_generator.state)
        action,info = controller.action(trace["packed_controller_states"][frame],frame,
                                        previous_action=previous,previous_vz=vz)
        mismatch = decision_mismatches(action,info,rows[frame],cost_atol=cost_atol,cost_rtol=cost_rtol)
        decision = {"frame":frame,"raw_action":action.tolist(),"solver":info,
            "full_plan_sha256":array_sha(controller.last_plan) if controller.last_plan is not None else None,
            "rng_state_before":rng_before,"rng_state_after":copy.deepcopy(controller.rng.bit_generator.state),
            "matched_available_archived_fields":not mismatch,"mismatched_fields":mismatch}
        with (Path(output)/"reconstructed_decisions.jsonl").open("a") as stream:
            stream.write(json.dumps(decision,allow_nan=False)+"\n")
        if mismatch:
            return matched,{"frame":frame,"fields":mismatch}
        matched.append(frame)
        if frame%10 == 0:
            print(json.dumps({"reconstructed_frame":frame,"candidate_evaluations":observer.total_candidates}),flush=True)
    return matched,None


class ObservedPredictor:
    """External delegating observer; no extra score, random draw or retry."""
    def __init__(self, base, output):
        self.base, self.output = base, Path(output)
        self.frame, self.frame_calls, self.total_candidates = -1, 0, 0
        self.native_calls_started, self.completed_native_substeps = 0, 0
        self.first_exception = None
        self.phase, self.current = "idle", None
        for worker in base.workers.values():
            original = worker.rollout
            def observed_native(*args, _original=original, **kwargs):
                self.phase = "native_transition"
                self.native_calls_started += 1
                try:
                    result = _original(*args, **kwargs)
                except Exception as error:
                    self.capture_exception(error)
                    raise
                self.completed_native_substeps += len(self.current["plans"])*30*4
                self.phase = "cost_adapter_objective"
                return result
            worker.rollout = observed_native

    def initial_candidate(self, *args, **kwargs):
        return self.base.initial_candidate(*args, **kwargs)

    def begin(self, frame):
        self.frame, self.frame_calls = frame, 0

    def capture_exception(self, error):
        if self.first_exception is not None:
            return
        current = self.current
        self.first_exception = {"frame":self.frame, "frame_call_index":self.frame_calls,
            "phase":self.phase, "exception_type":type(error).__name__, "message":str(error),
            "mu":current["mu"], "candidate_count":len(current["plans"]),
            "plan_sha256":array_sha(current["plans"]), "state_sha256":array_sha(current["state"]),
            "worker":worker_status(self.base.workers[current["mu"]]),
            "exact_historical_batch_independently_proven":False}
        np.savez_compressed(self.output/"first_exception_inputs.npz", state=current["state"],
            plans=current["plans"], mu=np.array(current["mu"]), previous_action=current["previous_action"],
            previous_vz=np.array(current["previous_vz"]), reference_index=np.array(self.frame))
        dump(self.output/"first_exception.json",self.first_exception)

    def rollout_batch(self, state, plans, start_index, *, mu, previous_action, previous_vz):
        if start_index != self.frame or not 0 <= self.frame <= FINAL_FRAME:
            raise ReconstructionStop("Prediction phase exceeds fixed reconstruction scope")
        if np.asarray(plans).ndim != 3 or plans.shape[1:] != (30,8) or len(plans)<1:
            raise ReconstructionStop("Prediction plan exceeds fixed H30/eight-action contract")
        if self.total_candidates+len(plans) > MAX_CANDIDATE_EVALUATIONS:
            raise ReconstructionStop("Declared reconstruction candidate budget exceeded")
        self.frame_calls += 1
        self.total_candidates += len(plans)
        self.current = {"state":np.asarray(state).copy(),"plans":np.asarray(plans).copy(),"mu":float(mu),
                        "previous_action":np.asarray(previous_action).copy(),"previous_vz":float(previous_vz)}
        record = {"frame":self.frame,"frame_call_index":self.frame_calls,"mu":float(mu),
                  "shape":list(plans.shape),"plan_sha256":array_sha(plans),"state_sha256":array_sha(state)}
        self.phase = "cost_adapter_input"
        try:
            costs = self.base.rollout_batch(state,plans,start_index,mu=mu,
                        previous_action=previous_action,previous_vz=previous_vz)
            record["costs"] = [float(x) if np.isfinite(x) else None for x in costs]
            return costs
        except Exception as error:
            self.capture_exception(error)
            record["exception_type"],record["message"] = type(error).__name__,str(error)
            raise
        finally:
            with (self.output/"prediction_calls.jsonl").open("a") as stream:
                stream.write(json.dumps(record,allow_nan=False)+"\n")


def prepare(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Preserve existing reconstruction/preparation: "+str(output))
    episode = Path(args.episode)
    for name, digest in EPISODE_SHA256.items():
        if sha(episode/name) != digest:
            raise ValueError("Diagnostic is bound to original I6 episode: "+name)
    report = json.loads((episode/"report.json").read_text())
    rows = json.loads((episode/"solver_log.json").read_text())
    if sha(episode/"snapshot_manifest.json") != SNAPSHOT_MANIFEST_SHA256:
        raise ValueError("Original source-manifest identity differs")
    manifest = json.loads((episode/"snapshot_manifest.json").read_text())
    if manifest["archive_sha256"] != ARCHIVE_SHA256 or manifest["safe_regular_member_count"] != 15:
        raise ValueError("Original source-archive contract differs")
    if (report["reference_id"] != "drift_drl_traj4" or report["prediction_mu_scenarios"] != [.7]
            or report["prediction_receives_plant_mu"] or report["effective_cem_config"]["seed"] != 42
            or report["effective_cem_config"]["horizon_steps"] != 30
            or report["effective_cem_config"]["iterations"] != 6
            or report["effective_cem_config"]["control_blocks"] != 5
            or report["effective_cem_config"]["candidates"] != 64):
        raise ValueError("Original planning configuration differs")
    for path, digest, role in ((args.model,report["mujoco_model_sha256"],"model"),
                               (args.motion,report["motion_sha256"],"motion"),
                               (args.contract,source_hash(report,"config/nominal_actuator_contract.json"),"actuator")):
        if sha(path) != digest:
            raise ValueError("Original "+role+" bytes differ")
    with np.load(episode/"rollout.npz",allow_pickle=False) as data:
        trace = {key:data[key].copy() for key in data.files}
    validate_recorded_inputs(trace,rows)
    helper_bytes = {}
    for relative in HELPERS:
        path = Path(args.source_root)/relative
        if path.is_symlink() or sha(path) != source_hash(report,relative):
            raise ValueError("Original helper bytes differ: "+relative)
        helper_bytes[relative] = path.read_bytes()
    output.mkdir(parents=True)
    sources = extract_snapshot(args.snapshot_archive,output/"original_snapshot",manifest["source_sha256"])
    for relative,data in helper_bytes.items():
        path = output/"original_helpers"/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    metadata = {"schema_version":1,"kind":"offline_original_isaac_I6_planning_prefix_reconstruction",
        "status":"prepared_no_planning","new_isaac_app":False,"new_plant_episode":False,
        "planning_mujoco_physics_executed":False,"new_build":False,"new_remote_upload":False,
        "preparation_status":"prepared_no_planning","planning_started":False,"planning_steps":0,
        "planning_steps_semantics":"Completed native batches only; a failed native call may have partially stepped and is not included. Zero before native construction is exact.",
        "fixed_frame_range_inclusive":[0,FINAL_FRAME],"max_candidate_evaluations":MAX_CANDIDATE_EVALUATIONS,
        "max_predicted_physics_substeps":MAX_PREDICTED_SUBSTEPS,"archive_sha256":ARCHIVE_SHA256,
        "episode_sha256":EPISODE_SHA256,"script_sha256":sha(__file__),"source_sha256":sources,
        "helper_sha256":{k:hashlib.sha256(v).hexdigest() for k,v in helper_bytes.items()},
        "effective_cem_config":report["effective_cem_config"],"objective_contract":report["objective_contract"],
        "model_sha256":sha(args.model),"motion_sha256":sha(args.motion),"contract_sha256":sha(args.contract),
        "python_version":platform.python_version(),"numpy_version":np.__version__,
        "remote_numpy_current_query_not_historical_proof":args.remote_numpy_current_query,
        "cost_guard":{"atol":args.cost_atol,"rtol":args.cost_rtol,"fixed_before_reconstruction":True},
        "action_guard":"Both raw float64 selected action and actual float32 cast must match exactly",
        "missing_historical_evidence":"Original FULL warm plans, per-candidate costs and elite indices were not saved; prefix consistency is not independent historical batch-byte proof.",
        "started_utc":datetime.now(timezone.utc).isoformat()}
    dump(output/"preparation.json",metadata)
    return report,rows,trace,metadata


def initialize_predictor(args, report, output):
    """All imports/model/provider preflight, still before native construction."""
    if args.native_library is None:
        raise ValueError("Planning requires the explicitly downloaded original native binary")
    library = Path(args.native_library).resolve()
    if sha(library) != NATIVE_SHA256 or sha(library.with_suffix(".json")) != NATIVE_MANIFEST_SHA256:
        raise ValueError("Use exact original remote native binary and untouched build manifest")
    scripts = output/"original_snapshot/scripts"
    modules = ("nmpc_shooting","nmpc_native_batch","nmpc_batch_objective","nmpc_tracking_objective",
               "run_nmpc_tracking","run_nmpc_tracking_isaac","classical_tracking","controller_references")
    if any(name in sys.modules for name in modules):
        raise RuntimeError("Use a fresh Python process; refusing preloaded controller modules")
    sys.path.insert(0,str(scripts))
    loaded = {name:importlib.import_module(name) for name in modules}
    for name,module in loaded.items():
        if Path(module.__file__).resolve() != scripts/(name+".py"):
            raise RuntimeError("A non-archived module shadowed the immutable snapshot: "+name)
    cfg = json.loads((output/"original_snapshot/config/nmpc_height_smooth_residual.json").read_text())
    profile = json.loads((output/"original_snapshot/config/car_drift_height_smooth_task.json").read_text())
    contract = json.loads(Path(args.contract).read_text())
    provider = loaded["run_nmpc_tracking_isaac"].SourceModelProvider(output/"original_helpers",args.model,args.motion,contract,cfg,(.7,))
    predictor = loaded["run_nmpc_tracking"].MujocoTrackingPredictor(provider,args.model,profile,(.7,))
    predictor.objective = loaded["nmpc_tracking_objective"].HeightSmoothTrackingObjective(profile,dt=.02,
                          soft_joint_limits=np.asarray(report["runtime_soft_joint_pos_limits"]))
    if predictor.objective.contract() != report["objective_contract"]:
        raise ValueError("Reconstructed objective contract differs")
    return loaded,predictor,library


def reconstruct(args):
    report,rows,trace,status = prepare(args)
    output = Path(args.output_dir).resolve()
    if args.prepare_only:
        return status
    native = None
    observer = None
    matched = []
    started = time.perf_counter()
    status.update(status="initializing",initialization_completed=False)
    dump(output/"report.json",status)
    try:
        loaded,predictor,library = initialize_predictor(args,report,output)
        status.update(native_library_sha256=sha(library),native_build_manifest_sha256=sha(library.with_suffix(".json")))
        native = loaded["nmpc_batch_objective"].NativeBatchTrackingPredictor(predictor,library)
        original_worker = report["prediction_backend_metadata"]["workers"]["0.7"]
        worker = native.workers[.7]
        for field in ("source_sha256","mujoco_version","header_version","mujoco_header_sha256",
                      "mujoco_library_sha256","native_library_sha256","model_snapshot_sha256"):
            if worker.metadata[field] != original_worker[field]:
                raise ReconstructionStop("Original worker/model identity differs: "+field)
        status["native_worker_metadata"] = copy.deepcopy(worker.metadata)
        observer = ObservedPredictor(native,output)
        controller = loaded["nmpc_shooting"].ShootingNMPC(observer,
            loaded["nmpc_shooting"].CEMConfig(**report["effective_cem_config"]),
            action_low=-np.ones(8),action_high=np.ones(8),mu_scenarios=(.7,))
        status.update(status="reconstructing",initialization_completed=True,planning_started=True)
        dump(output/"report.json",status)
        matched,mismatch = replay_prefix(controller,observer,trace,rows,output,
                                         cost_atol=args.cost_atol,cost_rtol=args.cost_rtol)
        if mismatch is not None:
            status.update(status="stopped_at_first_mismatch",first_mismatch=mismatch)
            raise ReconstructionStop("First mismatch at frame "+str(mismatch["frame"])+": "+", ".join(mismatch["fields"]))
        status["status"] = "matched_available_prefix_with_observed_exception" if observer.first_exception is not None else "matched_available_prefix_without_exception"
    except ReconstructionStop as error:
        if status["status"] != "stopped_at_first_mismatch":
            status["status"] = "stopped_by_identity_or_budget_guard"
        status["stop_reason"] = str(error)
    except BaseException as error:
        status.update(status="failed_diagnostic" if status["planning_started"] else "failed_initialization",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status.update(matched_frames=matched,elapsed_s=time.perf_counter()-started)
        if observer is not None:
            status.update(first_prediction_exception=observer.first_exception,
                          candidate_evaluations=observer.total_candidates,
                          maximum_possible_predicted_physics_substeps=observer.total_candidates*30*4,
                          planning_steps=observer.completed_native_substeps,
                          planning_mujoco_physics_executed=(True if observer.completed_native_substeps else
                              None if observer.native_calls_started else False))
        if native is not None:
            status["worker_before_outer_cleanup"] = worker_status(native.workers[.7])
            try:
                native.close()
            except Exception as cleanup_error:
                status["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                if not status["status"].startswith(("stopped", "failed")):
                    status["status"] = "failed_cleanup"
            status["worker_after_outer_cleanup"] = worker_status(native.workers[.7])
        status["prepared_source_hashes_unchanged"] = all(sha(output/"original_snapshot"/k)==v for k,v in status["source_sha256"].items())
        status["prepared_helper_hashes_unchanged"] = all(sha(output/"original_helpers"/k)==v for k,v in status["helper_sha256"].items())
        dump(output/"report.json",status)
    return status


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode",type=Path,default=ROOT/"outputs/nmpc_isaac_primary_nominal_residual_h30_b5_i6_20261016_v1")
    p.add_argument("--snapshot-archive",type=Path,default=ROOT/"outputs/nmpc_isaac_native_snapshot_20261016_v2.tar.gz")
    for name in ("source-root","model","motion","contract","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--native-library",type=Path)
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--remote-numpy-current-query",default=None,help="Read-only current observation, not proof of original runtime version")
    p.add_argument("--cost-atol",type=float,default=0.)
    p.add_argument("--cost-rtol",type=float,default=0.)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if not all(np.isfinite(v) and v>=0 for v in (arguments.cost_atol,arguments.cost_rtol)):
        raise SystemExit("Declare finite nonnegative cost tolerances before execution")
    result = reconstruct(arguments)
    print(json.dumps({"status":result["status"],"output":str(arguments.output_dir)}),flush=True)
    if result["status"].startswith(("stopped", "failed")):
        raise SystemExit(2)
