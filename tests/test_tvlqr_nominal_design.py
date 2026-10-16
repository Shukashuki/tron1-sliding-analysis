"""No simulation: synthetic donor bytes and exact quadratic identities."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/"scripts"))
from design_tvlqr_nominal import (JOINT_NAMES, augment_dynamics, design_objective_contract,
                                  load_nominal_donor, physical_state_cost,
                                  quadratic_stage_cost, sha)


@pytest.fixture
def donor(tmp_path):
    files = [tmp_path/name for name in ("model.xml", "motion.npz", "policy_contract.json")]
    for index, path in enumerate(files):
        path.write_bytes(f"synthetic non-executable input {index}".encode())
    directory = tmp_path/"donor"
    directory.mkdir()
    qpos = np.zeros((1397, 15)); qpos[:, 3] = 1.
    qpos[:, 0] = np.arange(1397)*.005
    actions = np.tile(np.linspace(-.3, .3, 8).astype(np.float32).astype(float), (349, 1))
    np.savez_compressed(directory/"rollout.npz", qpos=qpos, qvel=np.zeros((1397, 14)), actions=actions,
                        time_s=np.arange(1397)*.005, physics_dt_s=.005, policy_dt_s=.02,
                        action_joint_names=np.array(JOINT_NAMES), held_action=np.vstack((np.zeros(8), np.repeat(actions, 4, axis=0))),
                        reference_frame_for_action=np.r_[0, np.repeat(np.arange(349), 4)])
    solver = [{"frame": k, "action": action.tolist(), "solver": {"scenario_evaluations": 17, "horizon_steps": 30}}
              for k, action in enumerate(actions)]
    (directory/"solver_log.json").write_text(json.dumps(solver))
    report = dict(kind="full_body_height_smooth_cem_nmpc", engine="MuJoCo", status="completed",
                  reference_id="drift_drl_traj4", completed_policy_steps=349, evaluated_policy_steps=349,
                  plant_mu=.7, prediction_mu_scenarios=[.7], learned_policy_used=False,
                  root_state_prescribed_after_reset=False, external_root_forces_used=False,
                  truncated_smoke=False, model_generated_nominal_plan={"enabled": False},
                  rollout_sha256=sha(directory/"rollout.npz"), solver_log_sha256=sha(directory/"solver_log.json"),
                  model_sha256=sha(files[0]), motion_sha256=sha(files[1]),
                  source_sha256={str(files[2]): sha(files[2])}, wall_s=123., effective_cem_config={"horizon_steps": 30})
    (directory/"report.json").write_text(json.dumps(report))
    return directory, files, report


def test_donor_uses_350_boundary_states_and_actual_controls_with_provenance(donor):
    directory, files, _ = donor
    states, actions, previous, identity = load_nominal_donor(directory, *files)
    assert states.shape == (350, 29) and actions.shape == previous.shape == (349, 8)
    np.testing.assert_allclose(states[:, 0], np.arange(350)*.02)
    np.testing.assert_array_equal(previous[0], np.zeros(8))
    np.testing.assert_array_equal(previous[1:], actions[:-1])
    np.testing.assert_array_equal(actions, actions.astype(np.float32).astype(float))
    assert identity["generation_budget"]["scenario_evaluations"] == 349*17
    assert identity["generation_budget"]["predicted_physics_substeps"] == 349*17*30*4
    assert identity["archive_sha256"]["report.json"] == sha(directory/"report.json")
    assert identity["learned_policy_used"] is False


@pytest.mark.parametrize("field,value", [("learned_policy_used", True), ("plant_mu", .3),
                                         ("prediction_mu_scenarios", [.3, .7]),
                                         ("root_state_prescribed_after_reset", True),
                                         ("completed_policy_steps", 348),
                                         ("model_generated_nominal_plan", {"enabled": True})])
def test_donor_rejects_nonmatching_generation_contract(donor, field, value):
    directory, files, report = donor
    report[field] = value
    (directory/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_nominal_donor(directory, *files)


@pytest.mark.parametrize("target", ["rollout.npz", "solver_log.json", "model", "motion", "contract"])
def test_donor_input_byte_bindings_cannot_be_silently_changed(donor, target):
    directory, files, _ = donor
    path = directory/target if target.endswith(("npz", "json")) else files[["model", "motion", "contract"].index(target)]
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError):
        load_nominal_donor(directory, *files)


def test_donor_rejects_raw_float64_controls_even_when_archive_hash_is_updated(donor):
    directory, files, report = donor
    with np.load(directory/"rollout.npz", allow_pickle=False) as archive:
        trace = {name: archive[name].copy() for name in archive.files}
    trace["actions"][0, 0] = .123456789012
    np.savez_compressed(directory/"rollout.npz", **trace)
    report["rollout_sha256"] = sha(directory/"rollout.npz")
    (directory/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="applied-action"):
        load_nominal_donor(directory, *files)


def test_physical_weights_are_named_and_have_no_absolute_height_or_vz_tracking():
    vids = np.array([7, 6, 9, 8, 11, 10, 13, 12])
    diagonal = np.diag(physical_state_cost(vids, JOINT_NAMES))
    assert diagonal[2] == diagonal[16] == 0
    np.testing.assert_array_equal(diagonal[:6], [400., 400., 0., 500., 500., 120.])
    np.testing.assert_array_equal(diagonal[14:20], [20., 20., 0., 5., 5., 5.])
    np.testing.assert_array_equal(diagonal[vids[:6]], np.full(6, 5.))
    np.testing.assert_array_equal(diagonal[vids[6:]], np.zeros(2))
    np.testing.assert_array_equal(diagonal[14+vids[6:]], [.005, .005])
    with pytest.raises(ValueError):
        physical_state_cost(np.arange(8), JOINT_NAMES)


def test_augmented_previous_action_memory_retains_affine_physical_defect():
    A, B, defect = np.eye(28), np.arange(224).reshape(28, 8), np.arange(28.)
    aa, bb, dd = augment_dynamics(A, B, defect)
    np.testing.assert_array_equal(aa[:28, :28], A)
    np.testing.assert_array_equal(aa[28:], np.zeros((8, 36)))
    np.testing.assert_array_equal(bb[:28], B)
    np.testing.assert_array_equal(bb[28:], np.eye(8))
    np.testing.assert_array_equal(dd, np.r_[defect, np.zeros(8)])


def test_stage_quadratic_exactly_matches_rate_and_linearized_vertical_smoothness():
    rng = np.random.default_rng(42)
    Qphysical = physical_state_cost(np.arange(6, 14), JOINT_NAMES)
    action, previous, gx, gu = rng.normal(size=8), rng.normal(size=8), rng.normal(size=28), rng.normal(size=8)
    defect = .031
    Q, R, N, q, r = quadratic_stage_cost(Qphysical, action, previous, gx, gu, defect)
    for _ in range(10):
        z, v = rng.normal(size=36), rng.normal(size=8)
        quadratic = .5*z@Q@z + .5*v@R@v + z@N@v + q@z + r@v
        direct = (.5*z[:28]@Qphysical@z[:28] + .5*v@np.diag([10.]*6+[.5]*2)@v
                  + .1*np.sum((action-previous+v-z[28:])**2)
                  + 10*(defect+gx@z[:28]+gu@v)**2
                  - .1*np.sum((action-previous)**2) - 10*defect**2)
        assert quadratic == pytest.approx(direct, rel=1e-12, abs=1e-10)
    np.testing.assert_array_equal(Q, Q.T)
    np.testing.assert_array_equal(R, R.T)
    assert np.linalg.eigvalsh(np.block([[Q, N], [N.T, R]])).min() > -1e-10


def test_surrogate_and_height_free_objective_are_explicit_not_reward_parity():
    contract = design_objective_contract()
    assert contract["original_reward_matched"] is False
    assert contract["absolute_height_tracking"] is False
    assert contract["absolute_vz_tracking"] is False
    assert contract["sideslip_zero_target"] is False
    assert contract["flight_failure_gate"] is False
    json.dumps(contract, allow_nan=False)
