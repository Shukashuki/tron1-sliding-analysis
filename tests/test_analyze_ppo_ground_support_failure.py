"""CPU-only synthetic event/provenance tests and optional pinned-data regression."""
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_ppo_ground_support_failure as module
from test_analyze_ppo_contact_timestep import data


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")


@pytest.fixture
def cohort(data, monkeypatch):
    """Source pins are replaced ONLY with explicitly identified test fixtures."""
    root = data.tmp/"negative_cases"; root.mkdir()
    profile = {"contact_friction": 1., "reward_weights": {"grounded": .4},
        "height_regularization": {"mode": "vertical_acceleration_only"}}
    profile_path = root/"config/car_drift_height_smooth_task.json"; write_json(profile_path, profile)
    monkeypatch.setattr(module, "PROFILE_SHA256", module.smooth.sha256(profile_path))
    monkeypatch.setattr(module, "MOTION_SHA256", module.smooth.sha256(data.reference))
    specs = copy.deepcopy(module.CASE_SPECS)
    for case in module.CASE_ORDER:
        spec = specs[case]; spec["folder"] = case
        folder = root/case; folder.mkdir()
        source = data.root/("dt0p0025" if case == "baseline400" else "dt0p005")
        with np.load(source/"physics_trajectory.npz", allow_pickle=False) as z:
            raw = {k: z[k] for k in z.files}
        n = len(raw["time_s"])
        raw["nonwheel_ground_contact_force_norm_n"] = np.zeros((n, 1))
        if case == "ground200":
            mask = raw["time_s"] >= 2.
            raw["root_quaternion_wxyz"][mask, 0] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
            raw["nonwheel_ground_contact_force_norm_n"][raw["time_s"] >= 2.4] = 100.
        np.savez(folder/"physics_trajectory.npz", **raw)
        report = module.read_json(source/"report.json")
        report.update(checkpoint_sha256=spec["checkpoint_sha256"], task_profile=profile,
            evaluation_task_profile={**profile, "contact_friction": .7}, full_horizon_diagnostic=True,
            terminal_states_captured_before_auto_reset=True, domain_randomization_enabled=False, observation_noise_enabled=False,
            physics_trajectory_sha256=module.smooth.sha256(folder/"physics_trajectory.npz"))
        write_json(folder/"report.json", report)
        spec["raw_sha256"] = module.smooth.sha256(folder/"physics_trajectory.npz")
        spec["report_sha256"] = module.smooth.sha256(folder/"report.json")
    monkeypatch.setattr(module, "CASE_SPECS", specs)
    replacement = {"term": "drift_grounded", "weight": .4, "additional_reward_term": False,
        "reward_manager_dt_factor_applied_once": .02, "scale_fraction": .2, "new_formula": "synthetic disclosed fixture"}
    base = {key: {"same": key} for key in module.SHARED_TRAINING_KEYS}
    base.update(seed=42, num_envs=512, iterations=1000, num_steps_per_env=24,
        training_reward_overrides={"action_rate_l2": -.5})
    ground = copy.deepcopy(base)
    ground.update(contact_reward_replacement=replacement, comparison_baseline_action_rate_weight=-.5)
    def ablation(case):
        return {"status": "training_completed", "smoke_only": False, "initialization": "from_scratch_no_checkpoint_resume",
            "actual_environment_transitions": 12288000, "checkpoint_sha256": specs[case]["checkpoint_sha256"]}
    ground_ablation = ablation("ground200"); ground_ablation["reward_replacement"] = replacement
    training = {"baseline_manifest": base, "ground_manifest": ground,
        "baseline_ablation": ablation("baseline200"), "ground_ablation": ground_ablation}
    pins = {}
    for key, value in training.items():
        relative = "training/"+key+".json"; path = root/relative; write_json(path, value)
        pins[key] = (relative, module.smooth.sha256(path))
    monkeypatch.setattr(module, "TRAINING_FILES", pins)
    return SimpleNamespace(root=root, reference=data.reference, tmp=data.tmp)


def load(cohort): return module.load_comparison(cohort.root, cohort.reference)


def rebind_report(cohort, case, update):
    spec = module.CASE_SPECS[case]; path = cohort.root/spec["folder"]/"report.json"
    doc = module.read_json(path); update(doc); write_json(path, doc)
    spec["report_sha256"] = module.smooth.sha256(path)


def test_exact100ms_observed_run_and_posthoc_confirmation():
    t = np.arange(51)*.005; state = np.zeros(51, dtype=bool); state[2:22] = True
    result = module.persistent_events(t, state)
    event = result["first_persistent_event"]
    assert event["onset_s"] == .01 and event["end_s"] == .11
    assert event["confirmation_s"] == pytest.approx(.11) and event["native_samples"] == 20
    assert not event["duration_is_lower_bound"]
    state[21] = False
    assert module.persistent_events(t, state)["first_persistent_event"] is None


def test_censoring_no_extension_past_last_native_time():
    t = np.arange(51)*.005; state = np.ones(51, dtype=bool)
    event = module.persistent_events(t, state)["first_persistent_event"]
    assert event["left_window_censored"] and event["right_window_censored"] and event["duration_is_lower_bound"]
    assert event["end_s"] == t[-1] and event["observed_duration_s"] == .25


def test_brief_initial_sensor_nonwheel_transient_is_not_persistent():
    t = np.arange(100)*.0025; state = np.zeros(100, dtype=bool); state[0] = True
    result = module.persistent_events(t, state)
    assert result["first_true_sample_s"] == 0. and result["longest_observed_run_s"] == .0025
    assert result["first_persistent_event"] is None


@pytest.mark.parametrize("duration", [0., -.1, float("nan"), True])
def test_invalid_persistence_duration_rejected(duration):
    with pytest.raises(ValueError): module.persistent_events(np.arange(100)*.005, np.ones(100, dtype=bool), duration)


def test_matched_reward_shape_not_weight_increase_and_context_roles(cohort):
    raw, summary = load(cohort)
    assert summary["primary_matched_pair"] == ["baseline200", "ground200"]
    assert summary["context_only_cases"] == ["torque200", "baseline400"]
    assert not summary["matched_training"]["reward_weight_increased"]
    assert summary["matched_training"]["original_grounded_weight"] == summary["matched_training"]["candidate_grounded_weight"] == .4
    assert not summary["zero_contact_is_flight"] and not summary["height_or_contact_absence_failure_gate"]
    assert summary["cases"]["ground200"]["events_full"]["tilt_ge_30_deg"]["first_persistent_event"]["onset_s"] == 2.
    assert summary["cases"]["baseline200"]["events_full"]["tilt_ge_30_deg"]["first_persistent_event"] is None
    assert len(raw["baseline200"]["time_s"]) == 1397 and len(raw["baseline400"]["time_s"]) == 2793
    assert summary["cases"]["ground200"]["events_full"]["nonwheel_ground_normal_force_gt1N"]["first_persistent_event"]["onset_s"] == 2.4


def test_training_recipe_or_new_weight_mismatch_rejected(cohort):
    path = cohort.root/module.TRAINING_FILES["ground_manifest"][0]
    doc = module.read_json(path); doc["seed"] = 43; write_json(path, doc)
    relative, _ = module.TRAINING_FILES["ground_manifest"]
    module.TRAINING_FILES["ground_manifest"] = (relative, module.smooth.sha256(path))
    with pytest.raises(ValueError, match="training recipe differs: seed"): load(cohort)


def test_training_hash_mutation_fail_closed(cohort):
    path = cohort.root/module.TRAINING_FILES["ground_ablation"][0]
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="training metadata SHA"): load(cohort)


def test_raw_or_report_mutation_not_silently_rebound(cohort):
    path = cohort.root/"ground200"/"report.json"; path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="original report/raw SHA"): load(cohort)


@pytest.mark.parametrize("change", ["actor", "mu", "profile", "pd"])
def test_semantic_actor_mu_profile_pd_guards_even_after_test_fixture_repin(cohort, change):
    def mutate(doc):
        if change == "actor": doc["checkpoint_sha256"] = "0"*64
        elif change == "mu": doc["evaluation_contact_friction_override"]["evaluation_contact_friction"] = .8
        elif change == "profile": doc["task_profile"] = {"wrong": True}
        else: doc["policy_export"]["leg_kd"] = 5.
    rebind_report(cohort, "ground200", mutate)
    with pytest.raises(ValueError): load(cohort)


def test_native_counter_mutation_rejected_after_fixture_repin(cohort):
    spec = module.CASE_SPECS["ground200"]; path = cohort.root/spec["folder"]/"physics_trajectory.npz"
    with np.load(path) as z: raw = {k: z[k] for k in z.files}
    raw["simulation_step_counter"][10] += 1; np.savez(path, **raw)
    spec["raw_sha256"] = module.smooth.sha256(path)
    rebind_report(cohort, "ground200", lambda doc: doc.update(physics_trajectory_sha256=spec["raw_sha256"]))
    with pytest.raises(ValueError, match="Native counters"): load(cohort)


def test_inputs_immutable_and_full_tracking_not_extrapolated(cohort):
    paths = list(cohort.root.rglob("*.json"))+list(cohort.root.rglob("*.npz"))+[cohort.reference]
    before = {str(p): module.smooth.sha256(p) for p in paths}
    raw, summary = load(cohort)
    assert all(case["windows"]["full"]["tracking"]["available"] is False for case in summary["cases"].values())
    assert all(case["control_decisions"] == 349 for case in summary["cases"].values())
    assert {str(p): module.smooth.sha256(p) for p in paths} == before


def test_fresh_output_before_any_input_read(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "load_comparison", lambda *a: pytest.fail("Existing output must be rejected first"))
    args = module.parser().parse_args(["--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="fresh output"): module.run(args)


def test_publication_csv_censoring_and_optional_original_native_archive(cohort, monkeypatch):
    output = cohort.tmp/"publication"
    monkeypatch.setattr(module, "render_timeline", lambda raw, summary, path: path.write_bytes(b"synthetic test PNG placeholder"))
    args = module.parser().parse_args(["--input-root", str(cohort.root), "--reference", str(cohort.reference), "--output-dir", str(output), "--save-native"])
    summary = module.run(args)
    with (output/"events.csv").open() as stream: rows = list(csv.DictReader(stream))
    event = next(row for row in rows if row["case"] == "ground200" and row["condition"] == "tilt_ge_30_deg")
    assert float(event["onset_s"]) == 2. and float(event["confirmation_s"]) == 2.1
    assert event["right_window_censored"] == "True"
    with np.load(output/"native_states.npz", allow_pickle=False) as z:
        assert z["baseline200__time_s"].shape == (1397,) and z["baseline400__time_s"].shape == (2793,)
    assert json.loads((output/"summary.json").read_text())["matched_training"]["reward_weight_increased"] is False
    for artifact in summary["artifacts"].values(): assert module.smooth.sha256(output/artifact["file"]) == artifact["sha256"]
    with pytest.raises(ValueError, match="fresh output"): module.run(args)


def test_timeline_is_native_cpu_chart_with_fixed_geometry(cohort):
    raw, summary = load(cohort)
    output = cohort.tmp/"timeline_fixture.png"
    module.render_timeline(raw, summary, output)
    from PIL import Image
    with Image.open(output) as image:
        assert image.size == (1800, 1200)
    assert output.stat().st_size > 10000


REAL_INPUTS_PRESENT = (module.DEFAULT_REFERENCE.is_file()
    and all((module.ROOT/spec["folder"]/"physics_trajectory.npz").is_file() for spec in module.CASE_SPECS.values())
    and all((module.ROOT/path).is_file() for path, _ in module.TRAINING_FILES.values()))


@pytest.mark.skipif(not REAL_INPUTS_PRESENT, reason="Historical pinned private/local inputs are not present in a clean checkout")
def test_actual_pinned_negative_case_onsets_and_metrics_without_new_jobs():
    _, summary = module.load_comparison()
    ground = summary["cases"]["ground200"]; baseline = summary["cases"]["baseline200"]
    for threshold, onset in ((30, 2.335), (45, 2.425), (60, 2.485)):
        event = ground["events_full"][f"tilt_ge_{threshold}_deg"]["first_persistent_event"]
        assert event["onset_s"] == pytest.approx(onset) and event["right_window_censored"]
        assert event["confirmation_s"] == pytest.approx(onset+.1)
        assert baseline["events_full"][f"tilt_ge_{threshold}_deg"]["first_persistent_event"] is None
    assert ground["events_full"]["nonwheel_ground_normal_force_gt1N"]["first_persistent_event"]["onset_s"] == pytest.approx(3.3)
    assert ground["windows"]["core"]["tracking"]["xy_rmse_m"] == pytest.approx(.9443547141010037)
    assert baseline["windows"]["core"]["tracking"]["xy_rmse_m"] == pytest.approx(.056221388688688805)
    assert ground["windows"]["full"]["world_angular_xy_common10to100hz_rms_rad_s"] < baseline["windows"]["full"]["world_angular_xy_common10to100hz_rms_rad_s"]
    assert ground["windows"]["full"]["tilt_max_deg"] > 100.  # Lower fallen fullHF is NOT improved balance.
