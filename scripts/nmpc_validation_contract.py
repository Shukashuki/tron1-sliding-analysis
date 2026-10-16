"""Pure, local evidence gates for a future frozen NMPC friction validation.

No simulator, FK, inference, compilation, subprocess or network is invoked.
Hashes bind reported evidence and current bytes, not independent physical
truth. Freezing is an explicit caller action, never an inferred approval.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from analyze_controller_pair import audit_reports, _model_generated_nominal_plan_identity
from controller_trace import native_trace
from nmpc_shooting import CEMConfig
from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES


ENGINES = {"isaac": "Isaac/PhysX", "mujoco": "MuJoCo"}
FREEZE_KIND = "explicit_nominal_qualified_nmpc_validation_freeze"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    """Detached JSON-compatible representation; reject NaN/Infinity."""
    return json.loads(json.dumps(value, allow_nan=False))


def _digest(value):
    return hashlib.sha256(json.dumps(_json(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _number(value, label, *, nonnegative=False):
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value) and (not nonnegative or value >= 0), "Invalid " + label)
    return float(value)


def file_sha(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_bindings(paths: dict[str, Path]):
    """Hash existing files under portable logical roles, without storing paths."""
    _require(isinstance(paths, dict) and bool(paths), "Require input file roles")
    for role in paths:
        _require(isinstance(role, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:/-]*", role) is not None
                 and ".." not in role.split("/"), "Invalid logical input role")
    return {role: file_sha(path) for role, path in paths.items()}


def settings_contract(settings_path):
    settings = json.loads(Path(settings_path).read_text())
    _require(settings.get("schema_version") == 1 and settings.get("task") == "car_drift_height_smooth",
             "Unsupported NMPC settings schema/task")
    _require(settings.get("physics_dt_s") == .005 and settings.get("policy_dt_s") == .02,
             "Require the common 5 ms physics / 20 ms action clocks")
    config = CEMConfig(**settings["nmpc"])
    _require(type(config.seed) is int and config.seed >= 0, "Invalid CEM seed")
    for name in ("initial_std", "min_std"):
        value = np.asarray(getattr(config, name), dtype=float)
        _require(value.shape in ((), (8,)) and np.isfinite(value).all() and np.all(value > 0),
                 "Invalid CEM " + name)
    scenarios = settings.get("prediction_mu_scenarios")
    _require(isinstance(scenarios, list) and bool(scenarios), "Missing fixed prediction scenarios")
    scenarios = [_number(value, "prediction scenario", nonnegative=True) for value in scenarios]
    _require(len(set(scenarios)) == len(scenarios), "Duplicate prediction scenarios")
    return {"settings_file_sha256": file_sha(settings_path), "effective_cem_config": _json(asdict(config)),
            "prediction_mu_scenarios": scenarios, "physics_dt_s": .005, "control_dt_s": .02,
            "seed": config.seed}


def _array(raw, key, shape):
    value = np.asarray(raw[key])
    _require(value.shape == shape and value.dtype.kind in "biuf" and np.isfinite(value).all(),
             "Invalid native field: " + key)
    return value


def _material(report, engine, mu):
    def same(values):
        array = np.asarray(values, dtype=float)
        return array.size > 0 and np.isfinite(array).all() and np.allclose(array, mu, atol=1e-6, rtol=0)
    if engine == "mujoco":
        _require(same(report.get("plant_material_readback", {}).get("compiled_sliding_friction_range", [])),
                 "MuJoCo compiled material mismatch")
    else:
        _require(report.get("fixed_evaluation", {}).get("all_termination_terms_removed") is True,
                 "Isaac requires removal of all reset-triggering termination terms")
        material = report.get("material_audit", {})
        ground = material.get("ground_bound_materials", [])
        _require(same(material.get("expected_mu", [])) and same(material.get("robot_static_dynamic_range", []))
                 and bool(ground) and all(same(item.get("coefficients", [])[:2]) for item in ground),
                 "Isaac fixed material mismatch")


def _native_identity(report, engine):
    backend = report.get("prediction_backend")
    _require(backend in ("python", "native"), "Missing explicit prediction backend")
    identity = {"backend": backend, "native_library_sha256": None, "native_build_manifest_sha256": None}
    if backend == "native":
        metadata = report.get("prediction_backend_metadata" if engine == "isaac" else "native_prediction_backend", {})
        workers = metadata.get("workers", {})
        hashes = {worker.get("native_library_sha256") for worker in workers.values()}
        _require(bool(workers) and len(hashes) == 1 and all(_sha(value) for value in hashes),
                 "Missing/ambiguous native worker library identity")
        identity["native_library_sha256"] = hashes.pop()
        reported = report.get("native_library_sha256")
        _require(reported is None or reported == identity["native_library_sha256"], "Native library identities disagree")
        manifest = report.get("native_build_manifest_sha256")
        _require(manifest is None or _sha(manifest), "Invalid native build manifest identity")
        identity["native_build_manifest_sha256"] = manifest
    return identity


def model_nominal_plan_contract(report):
    """Normalize absent/OFF legacy declarations; preserve complete ON identity."""
    return _model_generated_nominal_plan_identity(report, report)


def _plan_bindings(plan, bindings, reference_spec):
    """Bind the optional JSON, numerical payload and actual loader source bytes."""
    roles = ("model_nominal_plan", "model_nominal_plan_actions", "source:scripts/model_nominal_plan.py")
    if not plan["enabled"]:
        _require(not any(role in bindings for role in roles),
                 "Disabled model nominal plan cannot retain enabled artifact/source bindings")
        return
    _require(all(_sha(bindings.get(role)) for role in roles),
             "Enabled model nominal plan requires metadata, actions and loader source bindings")
    metadata = plan["contract"]
    _require(bindings["model_nominal_plan"] == plan["artifact_sha256"]
             and metadata.get("actions_file") == "actions.npz"
             and metadata.get("actions_file_sha256") == bindings["model_nominal_plan_actions"],
             "Model nominal plan artifact/actions byte binding mismatch")
    identity = metadata.get("identity", {})
    _require(identity.get("reference_id") == reference_spec["reference_id"]
             and identity.get("reference_spec") == reference_spec
             and all(identity.get(name + "_sha256") == bindings[role] for name, role in (
                 ("motion", "motion"), ("robot_reference", "reference"), ("model", "model"),
                 ("actuator_contract", "contract"), ("task_profile", "task_profile"))),
             "Model nominal plan deployment identity differs from frozen inputs")


def _controller_contract(report, engine):
    """Stable per-engine declarations; exclude paths, timings and worker PIDs."""
    result = {name: _json(report.get(name)) for name in (
        "objective_contract", "packed_state_contract", "action_parameterization", "runtime_soft_joint_pos_limits",
        "contact_slip_definition", "contact_threshold_n", "action_order", "body_order",
        "mujoco_version", "prediction_mujoco_version", "torch_version")}
    result["model_generated_nominal_plan"] = model_nominal_plan_contract(report)
    sources = report.get("source_sha256", {})
    _require(isinstance(sources, dict) and bool(sources) and all(_sha(value) for value in sources.values()),
             "Require valid reported source SHA-256 identities")
    # Preserve duplicate values: two independently reported source files with
    # equal bytes must not collapse into one item. Host path names are not IDs.
    result["reported_source_sha256"] = sorted(sources.values())
    metadata = report.get("prediction_backend_metadata" if engine == "isaac" else "native_prediction_backend", {}) or {}
    result["backend"] = {name: _json(metadata.get(name)) for name in (
        "kind", "warnings", "whole_batch_warning_failure", "production_cem_wired",
        "plant_mu_used_for_prediction", "learned_initialization", "terminal_cost")}
    worker_fields = ("model_snapshot_sha256", "mujoco_version", "header_version", "abi_version",
        "source_sha256", "mujoco_header_sha256", "mujoco_library_sha256", "native_library_sha256",
        "body_names", "joint_names", "physics_dt_s", "control_dt_s", "reset_semantics", "measurement_semantics",
        "model_snapshot_owned_by_child", "fatal_errors_process_isolated", "ipc_included_in_latency")
    result["backend"]["workers"] = {str(mu): {name: _json(worker.get(name)) for name in worker_fields}
        for mu, worker in metadata.get("workers", {}).items()}
    return result


def episode_evidence(directory, engine, spec, expected_settings, plant_mu):
    """Validate a completed finite episode, including failed-controller fallbacks.

    Large XY error, no ground contact, height, tilt, beta, gamma and solver
    fallback counts never reject otherwise complete finite physical evidence.
    ``report`` and ``_directory`` are internal; do not publish them wholesale.
    """
    _require(engine in ENGINES, "Unknown episode engine")
    mu = _number(plant_mu, "plant mu", nonnegative=True)
    directory = Path(directory)
    files = {"report": directory / "report.json", "trace": directory / "rollout.npz",
             "solver": directory / "solver_log.json"}
    hashes = {name: file_sha(path) for name, path in files.items()}
    report = json.loads(files["report"].read_text())
    solver = json.loads(files["solver"].read_text())
    _require(isinstance(report, dict) and isinstance(solver, list), "Invalid episode report/solver log")
    _json(report); _json(solver)
    _require(report.get("kind") == ("receding_horizon_nmpc" if engine == "isaac" else "full_body_height_smooth_cem_nmpc"),
             "Require the explicit full-body NMPC report kind")
    _require(report.get("engine") == ENGINES[engine] and report.get("status") == "completed"
             and report.get("truncated_smoke") is False, "Require completed non-smoke engine episode")
    _require(report.get("reference_id") == spec["reference_id"] and report.get("role") == spec["role"]
             and report.get("stratum") == spec["stratum"] and report.get("reference_spec") == spec
             and report.get("motion_sha256") == spec["motion_npz_sha256"], "Episode reference identity mismatch")
    _require(report.get("rollout_sha256") == hashes["trace"] and report.get("solver_log_sha256") == hashes["solver"],
             "Episode trace/solver SHA-256 mismatch")
    _require(report.get("effective_cem_config") == expected_settings["effective_cem_config"]
             and report.get("prediction_mu_scenarios") == expected_settings["prediction_mu_scenarios"],
             "Episode differs from frozen CEM/settings/scenarios")
    _require(report.get("plant_mu") == mu and report.get("physics_dt_s") == .005
             and report.get("control_dt_s") == .02, "Episode plant coefficient/clock mismatch")
    for field in ("learned_policy_used", "tvlqr_used", "root_state_prescribed_after_reset",
                  "external_root_forces_used", "prediction_receives_plant_mu", "pose_height_failure_gate"):
        _require(report.get(field) is False, "Unsupported episode declaration: " + field)
    _require(report.get("action_replay", False) is False and report.get("old_qr_used", False) is False,
             "Unsupported replay/old objective")
    objective = report.get("objective_contract", {})
    _require(objective.get("absolute_height_target") is False and objective.get("reference_vertical_velocity_target") is False
             and objective.get("height_regularization", {}).get("mode") == "vertical_acceleration_only"
             and objective.get("dt_s") == .02 and objective.get("body_names") == list(BODY_NAMES)
             and objective.get("joint_names") == list(JOINT_NAMES)
             and len(objective.get("weights", {})) == 15, "Require the common height-free full-body objective")
    packed = report.get("packed_state_contract", {})
    _require(packed.get("dimension") == 111 and packed.get("qpos_dimension") == 15
             and packed.get("qvel_dimension") == 14 and packed.get("contact_force_history_shape") == [3, 9, 3],
             "Require the full physical/contact-memory packed state contract")
    _material(report, engine, mu)
    frames = spec["frames"]
    _require(type(frames) is int and frames > 0 and spec["fps"] == 50., "Invalid reference clock")
    samples = frames * 4 + 1
    _require(all(report.get(name) == frames for name in ("completed_policy_steps", "evaluated_policy_steps", "total_reference_policy_steps"))
             and np.isclose(report.get("duration_s", np.nan), frames * .02, atol=1e-9, rtol=0)
             and len(solver) == frames, "Incomplete full-reference episode")
    with np.load(files["trace"], allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    _require(all(value.dtype.kind not in "biufc" or np.isfinite(value).all() for value in raw.values()),
             "Nonfinite native numeric trace field")
    trace = native_trace(raw, "nmpc")
    _require(len(trace["time_s"]) == samples and trace["physics_dt_s"] == .005
             and np.allclose(trace["time_s"], np.arange(samples) * .005, atol=1e-9, rtol=0)
             and float(raw["policy_dt_s"]) == .02, "Incomplete native physical clock")
    _require(tuple(np.asarray(raw["body_names"]).tolist()) == BODY_NAMES, "Native body order mismatch")
    actions = _array(raw, "actions", (frames, 8))
    _require(np.max(np.abs(actions)) <= 1. + 1e-9
             and np.array_equal(actions, actions.astype(np.float32).astype(float)), "Invalid float32 normalized action interface")
    held = _array(raw, "held_action", (samples, 8))
    _require(np.array_equal(held, np.vstack((np.zeros((1, 8)), np.repeat(actions, 4, axis=0)))), "Held-action clock mismatch")
    phase = _array(raw, "reference_frame_for_action", (samples,))
    _require(np.array_equal(phase, np.r_[0, np.repeat(np.arange(frames), 4)]), "Held-reference phase mismatch")
    applied = _array(raw, "action_was_applied", (samples,))
    _require(applied.dtype.kind == "b" and np.array_equal(applied, np.r_[False, np.ones(samples - 1, dtype=bool)]),
             "Missing initial-only no-action sample")
    counts = trace["contact_count"]
    _require(np.all(counts >= 0) and np.array_equal(counts, np.round(counts)) and np.all(trace["contact_slip"] >= 0),
             "Invalid measured contact counts/slip")
    nonwheel = _array(raw, "nonwheel_ground_contact_count", (samples,))
    force = _array(raw, "nonwheel_ground_contact_force_norm_n", (samples,))
    _require(np.all(nonwheel >= 0) and np.array_equal(nonwheel, np.round(nonwheel)) and np.all(force >= 0),
             "Invalid measured nonwheel contacts")
    compute = _array(raw, "compute_s", (frames,))
    _require(np.all(compute >= 0), "Negative controller compute time")
    failures = fallbacks = 0
    for k, decision in enumerate(solver):
        _require(decision.get("frame") == k and decision.get("reference_frame_for_reward") == k
                 and np.array_equal(np.asarray(decision.get("action")), actions[k])
                 and decision.get("compute_s") == float(compute[k]), "Solver action/phase/clock mismatch")
        info = decision.get("solver", {})
        _require(type(info.get("optimization_failed")) is bool and type(info.get("fallback_used")) is bool,
                 "Missing solver failure/fallback diagnostics")
        failures += info["optimization_failed"]
        fallbacks += info["fallback_used"]
    _require(report.get("solver_failures") == failures, "Solver failure count mismatch")
    return {"engine": engine, "hashes": hashes, "frames": frames, "physics_samples": samples,
            "duration_s": frames * .02, "plant_mu": mu, "effective_cem_config": copy.deepcopy(report["effective_cem_config"]),
            "prediction_mu_scenarios": copy.deepcopy(report["prediction_mu_scenarios"]),
            "native_identity": _native_identity(report, engine),
            "controller_contract": _controller_contract(report, engine),
            "solver_diagnostics": {"optimization_failed": failures, "fallback_used": fallbacks},
            "report": report, "_directory": str(directory.resolve())}


def _bindings(bindings):
    _require(isinstance(bindings, dict) and all(_sha(value) for value in bindings.values()), "Invalid input SHA-256 bindings")
    _require(all(isinstance(role, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:/-]*", role) is not None
                 and ".." not in role.split("/") and not any(part in role for part in ("/home/", "/data/", "/private/"))
                 for role in bindings), "Invalid/private logical input role")
    aliases = {"motion": "motion_file", "reference": "reference_file", "model": "mujoco_model",
               "asset": "asset_path", "settings": "settings_file"}
    result = dict(bindings)
    for role, alias in aliases.items():
        if alias in result:
            _require(role not in result or result[role] == result[alias], "Conflicting input role alias")
            result[role] = result[alias]
    _require(all(role in result for role in ("motion", "reference", "model", "asset", "contract", "settings", "task_profile", "reference_catalog")),
             "Missing required frozen input roles")
    return result


def _source_bindings(report, engine, bindings):
    """Match caller-tracked implementation bytes to nominal declarations."""
    sources = report.get("source_sha256", {})
    common = {"scripts/nmpc_shooting.py", "scripts/nmpc_tracking_objective.py", "scripts/classical_tracking.py",
              "scripts/controller_references.py", "scripts/height_smoothness.py"}
    if model_nominal_plan_contract(report)["enabled"]:
        common.add("scripts/model_nominal_plan.py")
    required = common | {"scripts/run_nmpc_tracking_isaac.py" if engine == "isaac" else "scripts/run_nmpc_tracking.py"}
    if report.get("prediction_backend") == "native":
        required |= {"scripts/nmpc_native_batch.py", "scripts/nmpc_batch_objective.py", "native/nmpc_batch.cpp"}
    for role, digest in bindings.items():
        prefix, separator, relative = role.partition(":")
        if not separator or prefix not in ("source", "source_root", "mujoco_source"):
            continue
        if prefix == "source_root" and engine != "isaac" or prefix == "mujoco_source" and engine != "mujoco":
            continue
        matches = [value for name, value in sources.items() if name == relative or name.endswith("/" + relative)]
        mandatory = relative in required if prefix == "source" else True
        _require((not mandatory or bool(matches)) and all(value == digest for value in matches),
                 "Nominal source binding missing/mismatched for " + engine + ": " + role)


def make_freeze(*, reference_spec, settings_contract, bindings, nominal, pair_summary):
    """Explicitly freeze only a complete, matching two-engine nominal XY pass.

    This validates existing files again; it never creates missing qualification
    evidence, runs an analyzer, or writes the returned document to disk.
    """
    bound = _bindings(bindings)
    _require(bound["motion"] == reference_spec["motion_npz_sha256"]
             and bound["reference"] == reference_spec["robot_reference_npz_sha256"]
             and bound["settings"] == settings_contract["settings_file_sha256"], "Freeze reference/settings byte binding mismatch")
    _require(isinstance(nominal, dict) and set(nominal) == set(ENGINES), "Both actual nominal engine episodes are required")
    checked = {}
    for engine, evidence in nominal.items():
        _require(isinstance(evidence, dict) and "_directory" in evidence, "Nominal evidence must come from actual episode files")
        checked[engine] = episode_evidence(evidence["_directory"], engine, reference_spec, settings_contract, .7)
        _require(checked[engine] == evidence, "Nominal evidence changed after capture")
        report = evidence["report"]
        _require(report.get("mujoco_model_sha256" if engine == "isaac" else "model_sha256") == bound["model"],
                 "Nominal model binding mismatch")
        if engine == "isaac":
            _require(report.get("asset_sha256") == bound["asset"], "Nominal Isaac asset binding mismatch")
        source_hashes = report.get("source_sha256", {})
        _require(isinstance(source_hashes, dict)
                 and all(bound[role] in source_hashes.values() for role in ("contract", "settings", "task_profile")),
                 "Nominal reported source identities do not bind current actuator/settings/task-profile files")
        _source_bindings(report, engine, bound)
        identity = evidence["native_identity"]
        for suffix, key in (("native_library", "native_library_sha256"), ("native_build_manifest", "native_build_manifest_sha256")):
            role = engine + "_" + suffix
            if identity[key] is not None:
                _require(bound.get(role) == identity[key], "Nominal per-engine native binding mismatch: " + role)
    audit = audit_reports("NMPC", reference_spec, checked["isaac"]["report"], checked["mujoco"]["report"],
                          isaac_trace_sha256=checked["isaac"]["hashes"]["trace"],
                          mujoco_trace_sha256=checked["mujoco"]["hashes"]["trace"], model_sha256=bound["model"])
    plan = audit["nmpc_controller_identity"]["model_generated_nominal_plan"]
    _plan_bindings(plan, bound, reference_spec)
    if plan["enabled"]:
        _require(all(plan["artifact_sha256"] in evidence["report"]["source_sha256"].values()
                     for evidence in checked.values()), "Nominal reports do not bind the model plan JSON source")
    summary = pair_summary
    _require(summary.get("kind") == "offline_single_controller_fixed_mu_engine_pair" and summary.get("algorithm") == "NMPC"
             and summary.get("reference_spec") == reference_spec and summary.get("reference_id") == reference_spec["reference_id"]
             and summary.get("fixed_mu") == .7 and summary.get("integration_only") is False
             and summary.get("core_coverage_complete") is True and summary.get("eligible_for_controller_ranking") is True
             and summary.get("pooled_success_statistics") is False and summary.get("height_or_flight_failure_gate") is False
             and summary.get("heading_offset_is_mismatch") is False, "Require the matching full nominal common-pair summary")
    expected = {"motion_file": bound["motion"], "reference_file": bound["reference"], "model": bound["model"]}
    for engine in ENGINES:
        expected[engine + "_trace"] = checked[engine]["hashes"]["trace"]
        expected[engine + "_report"] = checked[engine]["hashes"]["report"]
    _require(summary.get("input_sha256") == expected, "Nominal pair summary input hashes differ from actual qualification files")
    _require(summary.get("audit") == audit, "Nominal summary audit differs from the current common-pair audit")
    analysis_sources = summary.get("analysis_source_sha256", {})
    _require(isinstance(analysis_sources, dict) and _sha(analysis_sources.get("analyze_controller_pair.py"))
             and all(_sha(value) for value in analysis_sources.values()), "Missing nominal analyzer source identities")
    core = summary.get("windows", {}).get("core", {})
    for engine in ENGINES:
        window = core.get(engine, {})
        _require(window.get("coverage_complete") is True and window.get("large_xy_exploratory_screen") is False
                 and window.get("window_s") == reference_spec["evaluation_core_interval_s"], "Both nominal XY screens must pass with complete core coverage")
        rmse = _number(window.get("reference_xy_rmse_m"), "nominal core XY RMSE", nonnegative=True)
        peak = _number(window.get("reference_xy_peak_m"), "nominal core XY peak", nonnegative=True)
        _require(rmse <= .15 and peak <= .30, "Nominal XY screen flag contradicts its recorded metrics")
    return {"schema_version": 1, "kind": FREEZE_KIND, "status": "frozen_for_validation", "algorithm": "NMPC",
            "reference_spec": _json(reference_spec), "settings_contract": _json(settings_contract), "bindings": _json(bindings),
            "nominal_plant_mu": .7, "nominal": {engine: {key: _json(value) for key, value in evidence.items()
                if key not in ("report", "_directory")} for engine, evidence in checked.items()},
            "nominal_pair_summary_canonical_sha256": _digest(summary), "nominal_pair_input_sha256": expected,
            "nominal_analysis_source_sha256": _json(analysis_sources),
            "nominal_pair_audit": audit, "nominal_core_diagnostics": {engine: _json(core[engine]) for engine in ENGINES},
            "model_generated_nominal_plan": _json(plan),
            "nominal_posture_diagnostics": _json(summary.get("posture_diagnostics", {})),
            "qualification_gates": ["complete_two_engine_nominal", "matching_frozen_contract", "complete_core_xy_screen_only"],
            "height_flight_tilt_beta_or_numeric_speed_gate": False, "pooled_success_statistics": False,
            "evidence_scope": "File identities and reported provenance; supplied offline summary is not independently recomputed physics or an execution proof.",
            "freeze_document_integrity": "Explicit caller-selected record, not cryptographic authorship or permission for remote work."}


def validate_freeze(freeze, reference_spec, settings_contract, bindings):
    """Reject stale, development-only or mismatched freeze documents."""
    bound = _bindings(bindings)
    _require(isinstance(freeze, dict) and freeze.get("schema_version") == 1 and freeze.get("kind") == FREEZE_KIND
             and freeze.get("status") == "frozen_for_validation" and freeze.get("algorithm") == "NMPC", "Require an explicit NMPC validation freeze")
    _require(freeze.get("reference_spec") == reference_spec and freeze.get("settings_contract") == settings_contract
             and freeze.get("bindings") == bindings, "Freeze differs from current reference/settings/input bytes")
    _require(freeze.get("nominal_plant_mu") == .7 and freeze.get("height_flight_tilt_beta_or_numeric_speed_gate") is False
             and freeze.get("pooled_success_statistics") is False and set(freeze.get("nominal", {})) == set(ENGINES),
             "Freeze lacks two-engine nominal qualification")
    plan = model_nominal_plan_contract(freeze)
    _plan_bindings(plan, bound, reference_spec)
    for engine in ENGINES:
        evidence = freeze["nominal"][engine]
        _require(evidence.get("engine") == engine and evidence.get("plant_mu") == .7
                 and evidence.get("frames") == reference_spec["frames"]
                 and evidence.get("physics_samples") == reference_spec["frames"] * 4 + 1
                 and evidence.get("effective_cem_config") == settings_contract["effective_cem_config"]
                 and evidence.get("prediction_mu_scenarios") == settings_contract["prediction_mu_scenarios"]
                 and set(evidence.get("hashes", {})) == {"report", "trace", "solver"}
                 and all(_sha(value) for value in evidence["hashes"].values()), "Incomplete frozen nominal engine evidence")
        window = freeze.get("nominal_core_diagnostics", {}).get(engine, {})
        _require(window.get("coverage_complete") is True and window.get("large_xy_exploratory_screen") is False
                 and window.get("window_s") == reference_spec["evaluation_core_interval_s"],
                 "Frozen nominal XY qualification is missing")
        _require(_number(window.get("reference_xy_rmse_m"), "frozen XY RMSE", nonnegative=True) <= .15
                 and _number(window.get("reference_xy_peak_m"), "frozen XY peak", nonnegative=True) <= .30,
                 "Frozen XY qualification contradicts its metrics")
        identity = evidence.get("native_identity", {})
        _require(identity.get("backend") in ("python", "native") and isinstance(evidence.get("controller_contract"), dict),
                 "Missing frozen controller/backend contract")
        _require(model_nominal_plan_contract(evidence["controller_contract"]) == plan,
                 "Frozen engine model nominal plan differs from its bound shared identity")
        for suffix, key in (("native_library", "native_library_sha256"), ("native_build_manifest", "native_build_manifest_sha256")):
            if identity.get(key) is not None:
                _require(bound.get(engine + "_" + suffix) == identity[key], "Frozen per-engine native identity mismatch")
    expected = {"motion_file": bound["motion"], "reference_file": bound["reference"], "model": bound["model"]}
    for engine in ENGINES:
        expected[engine + "_trace"] = freeze["nominal"][engine]["hashes"]["trace"]
        expected[engine + "_report"] = freeze["nominal"][engine]["hashes"]["report"]
    _require(freeze.get("nominal_pair_input_sha256") == expected, "Frozen nominal file identities disagree")
    audit = freeze.get("nominal_pair_audit", {})
    _require(audit.get("fixed_mu") == .7 and audit.get("height_free_objective_and_evaluation") is True
             and audit.get("integration_only") is False
             and audit.get("nmpc_controller_identity", {}).get("effective_cem_config") == settings_contract["effective_cem_config"],
             "Missing frozen matching height-free nominal pair audit")
    _require(model_nominal_plan_contract(audit.get("nmpc_controller_identity", {})) == plan,
             "Frozen nominal pair model plan audit differs from its bound identity")
    _require(_sha(freeze.get("nominal_pair_summary_canonical_sha256")), "Missing nominal summary identity")


def validate_case_against_nominal(case_evidence, nominal_public_evidence):
    """Compare stable per-engine controller/model declarations, never outcomes."""
    for key in ("engine", "frames", "physics_samples", "effective_cem_config", "prediction_mu_scenarios",
                "native_identity", "controller_contract"):
        actual, nominal = case_evidence.get(key), nominal_public_evidence.get(key)
        if key == "controller_contract" and isinstance(actual, dict) and isinstance(nominal, dict):
            actual, nominal = copy.deepcopy(actual), copy.deepcopy(nominal)
            for contract in (actual, nominal):
                contract["model_generated_nominal_plan"] = model_nominal_plan_contract(contract)
        _require(key in case_evidence and actual == nominal,
                 "Validation case changed nominal controller contract: " + key)
