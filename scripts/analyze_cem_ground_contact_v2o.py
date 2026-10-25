"""Strict offline audit of a completed ACTUAL V2o CEM nominal episode.

Shares the frozen A metrics and unchanged nominal screen, not its reward or
input-identity gate. No simulation, policy inference, FK, tuning or native build.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import analyze_cem_ground_contact_v2 as base
import diagnose_ground_contact_windows_v2 as contact
import ground_contact_reward_v2o as definition
import analyze_cem_ground_contact_v2b as prior_b
import analyze_cem_ground_contact_v2c as prior_c
import analyze_cem_ground_contact_v2d as prior_d
import analyze_cem_ground_contact_v2e as prior_e
import analyze_cem_ground_contact_v2f as prior_f
import analyze_cem_ground_contact_v2g as prior_g
import analyze_cem_ground_contact_v2i as prior_i
import analyze_cem_ground_contact_v2k as prior_k
import run_cem_ground_contact_v2o as runner
from controller_trace import native_trace
from ppo_backend_400hz_contract import require, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "config/ppo_ground_contact_v2o_400hz_v1.json"
PROTOCOL_SHA256 = "99f40ae13ad1fce9f2fed37069d6f1c877e611ef5a128f578b4bcc9bee00de6b"
METRICS_SOURCE_SHA256 = "540f85cb960669d30ef212a4d1ab68c0c1c6abe8508505ac8103dedbe3b1a0ed"
LIBRARY_SHA256 = "1c4c0a66da7770a9579e6a19d0aded25a6656426f76e068c686a0ad977435095"
BASELINE_PINS = {
    "summary": "4cfa8cb34d2213908c2e90183275974ae5ae6bb2de49ac449849ab0cea605a7f",
    "report.json": "343f6270e8be135e2afd4f06df69d8d4f1273cca2cc651188a1198c81ab901da",
    "rollout.npz": "eff2b735b07b481e6ead32b76a716066f71815401f91231345dde84d451804be"}
V2A_PINS = {
    "report.json": "a1573ea6a2c97d9c1a6f132f14d1b0a3aab9abdd650606b5f73602c980061081",
    "rollout.npz": "b4a6dc3ee5e22977140c669f4a9041fe207abaa1ecd129814bc6d3343e01a8ae"}
V2B_ANALYZER_SHA256 = "ae6cb319af8e3e23e20532f0a172f2c61097a090ea12e2e030355912ced8c2ff"
V2B_PINS = {
    "report.json": "37f8df44d21223be2069027ed4b43288052a5704ea1ba34179ed0ab3ca09e32a",
    "rollout.npz": "fe67b59bfa238d81d0b2528f159ebfcc2f3c415f0dc70da45b2b403caaace319"}
V2C_ANALYZER_SHA256 = "5b8bb74b45816e16d65250039490c75ed1cce4d5b2bcacaee88fd0b8d5be2416"
V2C_PINS = {
    "report.json": "0a44a1478c442b101c561d3a260417f458bb87d862257d66e8e4a55e100d6be1",
    "rollout.npz": "99a4ce2666ecfcf2fe19505dfd5e9527d61e699a45a031636a80ce4c14c7c3ba"}
V2D_ANALYZER_SHA256 = "7008d5d9612c295596cd2e263f6bd1099147776d656d627bee6eddcf9f2792c8"
V2D_PINS = {
    "report.json": "9bc292a50448fa081768b22e27382b0f2719d9dd4eeaf97ded3be9d8cd177720",
    "rollout.npz": "a0ca0821e5e19415c8eb4fe9f0e4d021661a5eed3bdf2bef627a91d31204a114"}
V2E_ANALYZER_SHA256 = "d6f9323876fea532fa06daf9ee862cfc6a9fede80104aaee1a807a50eb147446"
V2E_PINS = {"report.json": "d4642d53ce134d397575e0fe7300341a9298f1944f3eb1e25af93d1414973f44",
    "rollout.npz": "7dd5952c8039e7e9a64207526f55803bcdca81b24a8879bd5ec1748fa6823220"}
V2F_ANALYZER_SHA256 = "64d421dd25294bf26074e5ec2a6a9f080379c757c0a9cd3a0da6586d2c26df72"
V2F_PINS = {"report.json": "9310d7d511b3aa7883c3550fa2aa415fd8ed5c1df71971754dcf150e9a0e578b",
    "rollout.npz": "b3d16b49357186d68f14ecd9623c76997bb3676548cd04dde671735041dfbd37"}
V2I_ANALYZER_SHA256 = "d47dc2972bf932babd506d04c31de006a5cf89570183c14864509eadfbf1e7b5"
V2I_PINS = {"report.json": "68829f93738030528ffb96dd9ef78fd2f4bf934ab4b999340439ba46003132ff",
    "rollout.npz": "655f17cfc1967116640a8242fe759d1a24ab4620999805a7b94963a61d9282ce"}
V2G_ANALYZER_SHA256 = "a8243fba2608821836d25916d20a9a6cd17c486d8e20aa43559db03f81f61da9"
V2G_PINS = {"report.json": "5e2a797328f62cba40516f3ecaa598be6708285fdab7670542e29c6ec3ee575b",
    "rollout.npz": "f03e4c66e9ff65e97121d5c8f29e365c8b144f55f1501b2ae100c8df054bfb16"}
V2K_ANALYZER_SHA256 = "4e6a6df6bc639032f393ed46ca1d22db4dad89c10ce6a51ea206969321168720"
V2K_PINS = {
    "report.json": "220edda86e0e4fab5bfbcf4b18864ce6229addf781a462f5daf96a14e1f3d6b9",
    "rollout.npz": "0542ab5d8c8e6a64a1b04d16118811fcec7cd1a87e651871373344a0c98374d7"
}
SOURCE_PINS = {
    "scripts/run_cem_ground_contact_v2o.py": "e7845059f6e885a4b47dc893a537942965cf0632d0ecc3a376a327c7ffeef70f",
    "config/ppo_cem_matched_reward_400hz_v1.json": "410180756a7d489a6f8181157d5993813b17b60f7e103449b9bfa81a25c0984d",
    "config/ppo_ground_contact_v2o_400hz_v1.json": "99f40ae13ad1fce9f2fed37069d6f1c877e611ef5a128f578b4bcc9bee00de6b",
    "scripts/ground_contact_reward_v2.py": "5794ceb435c6ddc9cc374fa5ebe7718edc99d4c3181ebeac111cafe128461bf5",
    "scripts/ground_contact_reward_v2b.py": "359997a5e4e1734f14ffd8c64a61bbaf0f8fb1d0dd6b274001a77689bb62ba6c",
    "scripts/ground_contact_reward_v2c.py": "758eba1c7234cd43b211262541fce0cab6ad91be8ae0f10a44714d037322ebfb",
    "scripts/ground_contact_reward_v2d.py": "1f6245ca3ca08483704d4f42a044ba38b8d09f01d81dc6e71bbf423d206898a7",
    "scripts/ground_contact_reward_v2f.py": "a9e561a09ca2f0cbcd6c412a8e1860817aeccaff50660e9796866996a844d3e1",
    "scripts/ground_contact_reward_v2g.py": "81056b23fbf2052f2e0630d1888aa6b71556027ca108ab3b08dd768a912e493a",
    "scripts/ground_contact_reward_v2i.py": "2d805c5d878541da7a7ccc1f6af421ebe4b425fb9006756081f786ae79bf5101",
    "scripts/ground_contact_reward_v2k.py": "dbea1148c00a394b990964f7e7b26d3a7e9bdbe75e43975b3e9e79e7e0ebe18f",
    "scripts/ground_contact_reward_v2o.py": "05643c45044cf282af9a34f019f13afa2bfec43357d3d03de427ea8a3ed54b31",
    "scripts/nmpc_batch_ground_contact_v2k_400hz.py": "d2ec7042e26b9ef9121cdd66d42d041f703073b724d231cd39a4d80e752c1923",
    "scripts/nmpc_batch_ground_contact_v2o_400hz.py": "d0c19ad6e5bab2d897922de1bd939f703c96cf39b53d5cdaabeefc6422b15535",
    "scripts/nmpc_native_ground_contact_400hz.py": "dccee6518aae5c4e20e513e27ccb27641174cd07836c1f951617c16398b5aa3c",
    "scripts/run_ppo_cem_matched_reward_400hz.py": "f083a62813e8e5c7bf7099d90e603eaa3eea7909e148cfb636e613b200177bea",
    "scripts/nmpc_shooting.py": "7c667ef0cbd32c58ce53882302e75f5d33487bcfea51c2f6ec3435d454a480ce",
    "scripts/nmpc_tracking_objective.py": "ff22eebb2b22b9af7dba114c64fcd10f6b0a90822559af8f32cbe6884015b860",
    "scripts/nmpc_batch_objective.py": "1c9da90ae66778a71ceb2e6bb9d0dcf7f0a2259fafee39b37cbe958b79178fae",
    "scripts/nmpc_batch_objective_400hz.py": "3292c7753b2bbd336cb4944f6d64d417c738dfba8f994e878024dd52f5e11246",
    "scripts/nmpc_native_batch_400hz.py": "9c07f5e3d88f123ca067f73ead43679eeb378c35ed0afe51b6375870f654f335",
    "scripts/run_nmpc_tracking.py": "bce84b92fba8d3ea2f44c6cd5e5d32abd65bed6e20c4f1beb8473bae9cfb1977",
    "scripts/classical_tracking.py": "3fcafdb6a16058280aa1e5a0ade6113f6963ddf3085c25ebe2ed6fbb30661c57",
    "scripts/eval_ppo_backend_400hz_mujoco.py": "9887aa0b4b7ef74845f47f8623356490a9990e4449e0b42f4a77b2b35a49c1e4",
    "scripts/render_ppo_beta_physx_to_mujoco_400hz.py": "9268618566845d8f0fba02e806ce33c9168512466b57f495cfe294ed28acdbee",
    "native/nmpc_batch_ground_contact_400hz.cpp": "e80e76ad68352da00aeb4cf7215fc68f127b6a953d1afee4ad65e3ba0178aef1"
}


def load_protocol(path=DEFAULT_PROTOCOL):
    require(sha256(path) == PROTOCOL_SHA256, "Frozen O protocol SHA differs")
    protocol = read_json(path)
    require(protocol["cohort_id"] == "tron1_ground_contact_v2o_any_wheel_400hz_v1"
            and protocol["ground_contact_reward_v2"] == definition.validate_settings(), "Wrong O formula contract")
    require(protocol["nominal_acceptance_before_training"] == read_json(base.DEFAULT_PROTOCOL)["nominal_acceptance_before_training"],
            "C must not relax A nominal acceptance")
    require(sha256(base.__file__) == METRICS_SOURCE_SHA256, "Frozen shared metrics/screen source differs")
    return protocol


def validate_metadata(report, experiment, protocol_path=DEFAULT_PROTOCOL):
    load_protocol(protocol_path)
    require(report.get("status") == experiment.get("status") == "completed"
            and report.get("smoke_only") is False and report.get("truncated_smoke") is False
            and report.get("completed_policy_steps") == 349 and report.get("native_samples") == 2793
            and report.get("completed_full_reference") is True, "Only terminal complete349-step O episodes may be scored")
    require(report.get("kind") == "400hz_CEM_ground_contact_reward_v2o_anywheel"
            and report.get("reward_definition_version") == "ground_contact_v2o_any_wheel"
            and report.get("protocol_sha256") == PROTOCOL_SHA256
            and experiment.get("kind") == "ground_contact_reward_v2o_CEM_nominal_only", "Wrong O identity/version/protocol")
    require(report.get("source_sha256") == experiment.get("source_sha256") == SOURCE_PINS,
            "Require exact complete frozen O source ledger")
    for name, digest in SOURCE_PINS.items():
        require(sha256(ROOT / name) == digest, "Executed frozen O source changed: " + name)
    require(report.get("condition") == "beta_on" and report.get("plant_mu") == .7
            and report.get("prediction_mu_scenarios") == [.7] and report.get("prediction_friction_known") is True
            and report.get("floor_filter_verified") is True
            and report.get("termination_or_reset_after_initialization") is False
            and report.get("height_contact_tilt_failure_gate") is False
            and (report.get("physics_dt_s"), report.get("policy_dt_s"), report.get("decimation")) == (.0025, .02, 8),
            "Wrong O material/floor/reset/native clock")
    require(report.get("ground_contact_reward_v2") == definition.formula_metadata()
            and report.get("ground_reward_external_weight") == .4
            and report.get("reward_term_count") == 15 and report.get("other_reward_terms_unchanged") == 14
            and report.get("actual_reward_uses_ground_contact_v2") is True, "N actual-stage reward contract differs")
    require(experiment.get("native_abi") == 3 and report.get("native_library_sha256") == LIBRARY_SHA256
            and report.get("native_backend", {}).get("native_abi") == 3, "Wrong frozen ABI3 worker")
    require(experiment.get("sample_size") == 1 and experiment.get("new_policy_training_or_inference") is False
            and experiment.get("automatic_retry") is False, "Wrong declared episode scope")
    require(report.get("CEM_config") == read_json(runner.nominal.DEFAULT_SETTINGS)["nmpc"]
            and report.get("discount_factor") == .99, "Frozen O optimizer/discount differs")


def load_completed(folder, protocol_path=DEFAULT_PROTOCOL):
    folder = Path(folder).resolve()
    names = ("report.json", "rollout.npz", "solver_log.json", "experiment.json")
    require(all((folder/name).is_file() and not (folder/name).is_symlink() for name in names),
            "Missing direct completed O files; prefixes/smokes are not analyzable")
    hashes = {name: sha256(folder/name) for name in names}
    report, experiment = read_json(folder/"report.json"), read_json(folder/"experiment.json")
    validate_metadata(report, experiment, protocol_path)
    require(experiment["report_sha256"] == hashes["report.json"]
            and report["rollout_sha256"] == hashes["rollout.npz"]
            and report["solver_log_sha256"] == hashes["solver_log.json"], "N terminal file SHA bindings differ")
    with np.load(folder/"rollout.npz", allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    decisions = json.loads((folder/"solver_log.json").read_text())
    require(isinstance(decisions, list), "N solver log must be a decision list")
    require(len(decisions) == 349 and raw["qpos"].shape == (2793, 15)
            and raw["actions"].shape == (349, 8), "Incomplete O states/actions/decisions")
    failures = sum(bool(row["solver"]["optimization_failed"]) for row in decisions)
    require(failures == report["solver_failures"], "Reported O solver failure count differs")
    reward_audit = runner.recompute_actual_rewards(raw, report, decisions)
    require(reward_audit == report["actual_reward_audit"], "Actual O 15-term reward recomputation differs")
    require(all(sha256(folder/name) == digest for name, digest in hashes.items()), "N completed files changed during loading")
    return raw, report, hashes, reward_audit


def load_original(reference):
    require(sha256(base.BASELINE_SUMMARY) == BASELINE_PINS["summary"], "Published original baseline summary changed")
    hashes = {name: sha256(base.BASELINE_FOLDER/name) for name in ("report.json", "rollout.npz")}
    require(all(hashes[name] == BASELINE_PINS[name] for name in hashes), "Original CEM baseline SHA changed")
    row = next(value for value in read_json(base.BASELINE_SUMMARY)["rows"] if value["id"] == "cem_beta_on")
    require(row["report_sha256"] == hashes["report.json"] and row["recording_sha256"] == hashes["rollout.npz"],
            "Original CEM public row binding differs")
    with np.load(base.BASELINE_FOLDER/"rollout.npz", allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    result = base.recorded_metrics(raw, reference)
    for name, published in (("core", "core"), ("whole_reference_overlap", "reference_overlap")):
        require(result["windows"][name]["tracking"] == row["windows"][published]["tracking"],
                "Original baseline metric recomputation differs")
    return result, hashes


def load_v2a(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(all(sha256(folder/name) == digest for name, digest in V2A_PINS.items()), "Optional A baseline SHA differs")
    raw, report, hashes, audit = base.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "A/N observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "A/N fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2a CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_O_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def load_v2b(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(sha256(prior_b.__file__) == V2B_ANALYZER_SHA256, "Frozen prior B analyzer changed")
    require(all(sha256(folder/name) == digest for name, digest in V2B_PINS.items()), "Optional B baseline SHA differs")
    raw, report, hashes, audit = prior_b.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "B/N observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "B/N fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2b CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_O_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def load_v2c(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(sha256(prior_c.__file__) == V2C_ANALYZER_SHA256, "Frozen prior C analyzer changed")
    require(all(sha256(folder/name) == digest for name, digest in V2C_PINS.items()), "Optional C baseline SHA differs")
    raw, report, hashes, audit = prior_c.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "C/N observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "C/N fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2c CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_O_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def load_v2d(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(sha256(prior_d.__file__) == V2D_ANALYZER_SHA256, "Frozen prior D analyzer changed")
    require(all(sha256(folder/name) == digest for name, digest in V2D_PINS.items()), "Optional D baseline SHA differs")
    raw, report, hashes, audit = prior_d.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "D/N observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "D/N fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2d CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_O_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def load_v2e(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(sha256(prior_e.__file__) == V2E_ANALYZER_SHA256, "Frozen prior E analyzer changed")
    require(all(sha256(folder/name) == digest for name, digest in V2E_PINS.items()), "Optional E baseline SHA differs")
    raw, report, hashes, audit = prior_e.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "E/N observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "E/N fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2e CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_O_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def load_v2f(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(sha256(prior_f.__file__) == V2F_ANALYZER_SHA256, "Frozen prior F analyzer changed")
    require(all(sha256(folder/name) == digest for name, digest in V2F_PINS.items()), "Optional F baseline SHA differs")
    raw, report, hashes, audit = prior_f.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "F/N observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "F/N fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2f CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_O_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def load_v2g(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(sha256(prior_g.__file__) == V2G_ANALYZER_SHA256, "Frozen prior G analyzer changed")
    require(all(sha256(folder/name) == digest for name, digest in V2G_PINS.items()), "Optional G baseline SHA differs")
    raw, report, hashes, audit = prior_g.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "G/N observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "G/N fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2g CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_O_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def load_v2i(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(sha256(prior_i.__file__) == V2I_ANALYZER_SHA256, "Frozen prior I analyzer changed")
    require(all(sha256(folder/name) == digest for name, digest in V2I_PINS.items()), "Optional I baseline SHA differs")
    raw, report, hashes, audit = prior_i.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "I/N observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "I/N fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2i CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_O_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def load_v2k(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(sha256(prior_k.__file__) == V2K_ANALYZER_SHA256, "Frozen prior K analyzer changed")
    require(all(sha256(folder/name) == digest for name, digest in V2K_PINS.items()), "Optional I baseline SHA differs")
    raw, report, hashes, audit = prior_k.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "K/N observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "K/N fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2k CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_O_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve analyses; fresh output directory required")
    protocol = load_protocol(args.protocol)
    raw, report, hashes, reward_audit = load_completed(args.input_dir, args.protocol)
    require(report["motion_sha256"] == sha256(runner.paired.DEFAULT_REFERENCE), "Locked reference SHA differs")
    reference = base.metrics.reference_data(runner.paired.DEFAULT_REFERENCE)
    old, old_hashes = load_original(reference)
    actual = base.recorded_metrics(raw, reference)
    trace = native_trace(raw, "nmpc")
    diagnosis = contact.analyze_contact(trace["time_s"], trace["position"], trace["com_velocity"], trace["quaternion"],
        raw["ground_normal_force_w"][..., 2], report["model_weight_n"], reference,
        nonwheel=raw.get("nonwheel_ground_contact_force_norm_n"), include_sensitivity=False)
    diagnosis.update(actual_recorded_reward_candidate="ground_contact_v2o",
        normalization_weight_source="SHA-bound O validation model_weight_n; actual native MuJoCo model weight",
        input_sha256=hashes, no_counterfactual_controller_or_PPO_performance=True)
    prior = None
    if getattr(args, "v2a_input_dir", None) is not None:
        prior = load_v2a(args.v2a_input_dir, reference, raw, report)
    previous_b = None
    if getattr(args, "v2b_input_dir", None) is not None:
        previous_b = load_v2b(args.v2b_input_dir, reference, raw, report)
    previous_c = None
    if getattr(args, "v2c_input_dir", None) is not None:
        previous_c = load_v2c(args.v2c_input_dir, reference, raw, report)
    previous_d = None
    if getattr(args, "v2d_input_dir", None) is not None:
        previous_d = load_v2d(args.v2d_input_dir, reference, raw, report)
    previous_e = None
    if getattr(args, "v2e_input_dir", None) is not None:
        previous_e = load_v2e(args.v2e_input_dir, reference, raw, report)
    previous_f = None
    if getattr(args, "v2f_input_dir", None) is not None:
        previous_f = load_v2f(args.v2f_input_dir, reference, raw, report)
    previous_g = None
    if getattr(args, "v2g_input_dir", None) is not None:
        previous_g = load_v2g(args.v2g_input_dir, reference, raw, report)
    previous_i = None
    if getattr(args, "v2i_input_dir", None) is not None:
        previous_i = load_v2i(args.v2i_input_dir, reference, raw, report)
    previous_k = None
    if getattr(args, "v2k_input_dir", None) is not None:
        previous_k = load_v2k(args.v2k_input_dir, reference, raw, report)
    screen = base.nominal_screen(actual, old, protocol)
    summary = {"kind": "CEM_ground_contact_V2o_ACTUAL_nominal_readonly_audit", "status": "completed_analysis",
        "input_directory": str(Path(args.input_dir).resolve()), "input_sha256": hashes,
        "protocol_sha256": sha256(args.protocol), "analyzer_sha256": sha256(__file__),
        "shared_metric_source_sha256": METRICS_SOURCE_SHA256,
        "contact_diagnostic_source_sha256": sha256(contact.__file__), "executed_source_sha256": copy.deepcopy(SOURCE_PINS),
        "sample_size_per_case": 1, "actual_O_reward_audit": reward_audit, "actual_V2o": actual,
        "original_grounded_baseline": old, "baseline_input_sha256": old_hashes,
        "baseline_summary_sha256": BASELINE_PINS["summary"], "optional_prior_actual_V2a": prior,
        "optional_prior_actual_V2b": previous_b, "optional_prior_actual_V2c": previous_c, "optional_prior_actual_V2d": previous_d,
        "optional_prior_actual_V2e": previous_e,
        "optional_prior_actual_V2f": previous_f,
        "optional_prior_actual_V2g": previous_g,
        "optional_prior_actual_V2i": previous_i,
        "optional_prior_actual_V2k": previous_k,
        "nominal_screen": screen,
        "runtime": {key: report[key] for key in ("solver_failures", "plant_warning_counts", "compute_ms_median_p95_max")},
        "no_new_dynamics_policy_inference_training_build_or_tuning": True,
        "overall_task_goal_complete_claimed": False,
        "limitations": ["ACTUAL closed-loop O episode only; not offline A-force sensitivity or a PPO result.",
            "N=1 per controller; no causal population effect or universal friction threshold.",
            "Reward definitions differ across original/A/B/C/D/E/F: episode reward sums are not a fair performance ranking.",
            "Any-wheel continuity requires actual Fz>5N; the other wheel may lift; zero force is not a geometric flight label.",
            "Acceptance criteria are frozen unchanged; no height/tilt/beta gate replaces independent XY tracking.",
            "Numerical CEM acceptance does not certify qualitative review, PPO/dual-engine results or overall task completion."]}
    arrays = {"time_s": trace["time_s"], "position": trace["position"], "velocity": trace["com_velocity"],
        "quaternion": trace["quaternion"], "ground_normal_Fz_n": raw["ground_normal_force_w"][..., 2],
        "evaluation_weight_n": np.array(report["model_weight_n"])}
    arrays.update({"reference_"+name: reference[name] for name in ("time_s", "position", "velocity")})
    if "nonwheel_ground_contact_force_norm_n" in raw:
        arrays["nonwheel_ground_force_norm_n"] = raw["nonwheel_ground_contact_force_norm_n"]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"native_contact_context.npz", **arrays)
    diagnosis["compact_sha256"] = sha256(output/"native_contact_context.npz")
    write_json(output/"contact_diagnostics.json", diagnosis)
    summary["contact_diagnostics_sha256"] = sha256(output/"contact_diagnostics.json")
    require(all(sha256(Path(args.input_dir)/name) == digest for name, digest in hashes.items()), "N inputs changed during analysis")
    write_json(output/"summary.json", summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--input-dir", type=Path, required=True)
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--v2a-input-dir", type=Path, help="Optional SHA-pinned prior ACTUAL A episode, not reward-sum ranking")
    result.add_argument("--v2b-input-dir", type=Path, help="Optional SHA-pinned prior ACTUAL B episode, not reward-sum ranking")
    result.add_argument("--v2c-input-dir", type=Path, help="Optional SHA-pinned prior ACTUAL C episode, not reward-sum ranking")
    result.add_argument("--v2d-input-dir", type=Path, help="Optional SHA-pinned prior ACTUAL D episode, not reward-sum ranking")
    result.add_argument("--v2e-input-dir", type=Path, help="Optional SHA-pinned prior ACTUAL E episode; no reward-sum ranking")
    result.add_argument("--v2f-input-dir", type=Path, help="Optional SHA-pinned prior ACTUAL F episode; no reward-sum ranking")
    result.add_argument("--v2g-input-dir", type=Path, help="Optional SHA-pinned prior ACTUAL G episode; no reward-sum ranking")
    result.add_argument("--v2i-input-dir", type=Path, help="Optional SHA-pinned actual parent I, not a reward-sum ranking")
    result.add_argument("--v2k-input-dir", type=Path, help="Optional SHA-pinned actual parent K; no reward-sum ranking")
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
