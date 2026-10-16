"""Asset-free contract checks for the sanitized terminal prefix diagnostic."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/"results/nmpc_i6_prefix_reconstruction_20261016_v1.json"
NOTE=ROOT/"results/2026-10-16-nmpc-prefix-reconstruction.md"


def publication():
    return json.loads(PUBLIC.read_text())


def test_prefix_publication_is_diagnostic_only_and_stops_at_first_mismatch():
    d=publication()
    assert d["status"]=="stopped_at_first_mismatch" and d["process_exit_code"]==2
    for key in ("is_new_controller_evaluation","new_plant_episode","new_isaac_app","new_tuning_trial",
                "new_remote_deployment","retry_or_resume_performed","original_prediction_failure_cause_identified"):
        assert d[key] is False
    assert d["scope"]=={"first_frame":0,"last_allowed_frame":137,"completed_decisions":138,
        "matched_prefix_first_frame":0,"matched_prefix_last_frame":136,"matched_prefix_decisions":137,
        "first_mismatch_frame":137,"continued_after_first_mismatch":False}
    assert d["matched_prefix_evidence"]["independent_historical_full_plan_byte_proof"] is False
    assert d["predeclared_guards"]["raw_selected_action"]=="bitwise_exact"
    assert d["predeclared_guards"]["applied_float32_action"]=="bitwise_exact"
    assert d["predeclared_guards"]["initial_selected_and_six_iteration_costs_atol"]==1e-12
    assert d["predeclared_guards"]["initial_selected_and_six_iteration_costs_rtol"]==1e-12
    assert d["predeclared_guards"]["tolerances_changed_after_observation"] is False
    assert d["matched_prefix_evidence"]["maximum_absolute_cost_difference"]==2.886579864025407e-15


def test_prefix_publication_preserves_failure_difference_not_successful_reproduction():
    d=publication();old=d["frame_137"]["archived"];new=d["frame_137"]["reconstructed"]
    assert old["candidate_evaluations"]==new["candidate_evaluations"]==387
    assert (old["finite_candidates"],old["failed_candidates"],old["failed_batch_prediction_calls"])==(323,64,1)
    assert (new["finite_candidates"],new["failed_candidates"],new["failed_batch_prediction_calls"])==(387,0,0)
    assert old["selected_model_cost"]==2.7299900332769544
    assert new["selected_model_cost"]==2.6327593627468584
    budget=d["planning_budget"]
    assert budget["actual_candidate_evaluations"]==budget["maximum_candidate_evaluations"]==53405
    assert budget["completed_predicted_physics_substeps"]==53405*30*4==6408600
    assert budget["prediction_calls"]==1241 and budget["prediction_exceptions_captured"]==0
    assert budget["worker_alive_before_normal_cleanup"] is True
    assert budget["worker_exit_code_after_normal_cleanup"]==0
    assert "not reproduced" in d["conclusion"] and "original failure cause" in d["conclusion"]


def test_prefix_publication_preserves_preflight_vs_first_planning_distinction():
    history=publication()["preflight_history"]
    assert history["v1"]["planning_steps"]==0
    assert history["v1"]["preserved_without_overwrite"] is True
    assert history["v1"]["included_as_planning_attempt"] is False
    assert history["v2"]["first_actual_planning_attempt"] is True
    assert history["v2"]["current_runtime_query_is_historical_version_proof"] is False


def test_prefix_publication_hashes_pin_original_not_updated_controller():
    hashes=publication()["identity_sha256"]
    assert all(re.fullmatch(r"[0-9a-f]{64}",value) for value in hashes.values())
    assert hashes["original_cem_source"]=="8d2ac598d9e0e2765f9566954ba3fd0f443d939d2a5c6566125cca18668b35bd"
    assert hashes["original_native_python_source"]=="475ef6fe1712783ea11af2cae64b5ff0d61321e6edee0a596f9977a7a69da8a1"
    assert hashes["original_native_binary"]=="ce676e8414d38beb0676e8529cd88ebfea980ef9d91494c052e21414c4f89ea5"
    assert hashes["compiled_worker_model"]=="28ebe2555a1f7511f882c5e62e914ece71bee6855a7ab4425e6fd614bfbe4b9e"
    assert hashes["reconstruction_report"]=="81a198de61e78a6d44c0318175238d3b08afb026053dda571fc6fbeb6249e053"
    assert publication()["source_audit"]=={"original_source_members":15,"original_historical_helpers":8,
                                           "all_source_and_helper_bytes_unchanged":True}


def test_prefix_publication_contains_no_private_payload_or_machine_details():
    d=publication()
    banned_keys={"pid","worker_pid","app_pid","compiler_command","working_directory","environment",
                 "qpos","qvel","state","packed_controller_states","plans","raw_action","actions",
                 "rng_state_before","rng_state_after","source_root","native_library_path"}
    def check(value):
        if isinstance(value,dict):
            assert not banned_keys.intersection(value)
            for item in value.values():check(item)
        elif isinstance(value,list):
            assert not any(isinstance(item,(int,float,list)) for item in value)
            for item in value:check(item)
        elif isinstance(value,str):
            assert not any(token in value for token in ("/home/","/data/","/tmp/","zengxiong@","14.103.233.29","117.50.226.213"))
    check(d)
    note=NOTE.read_text()
    assert "nmpc_i6_prefix_reconstruction_20261016_v1.json" in note
    assert "not reproduced" in note and "cause remains" in note
    assert "not independent proof" in note
    assert "first actual planning" in note and "missing SciPy" in note
