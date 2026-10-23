"""CPU-only force buffer/observer fixtures; no actor/physics/Isaac imports."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import eval_ppo_ground_force_api_probe_v2r2 as mod


def buffers(force=(2., 3., 4.), normal=(0., 0., 2.), start=2):
    f = np.zeros(32); n = np.zeros((32, 3)); n[:] = (0, 0, 1)
    f[start:start+len(force)] = force; n[start:start+len(force)] = normal
    return f, np.zeros((32, 3)), n, np.zeros(32), np.array([[len(force)]]), np.array([[start]])


def test_all_weak_points_sum_above_threshold_no_count_gate():
    total, count, start, length, active = mod.point_sum(buffers())
    assert np.array_equal(total, [0, 0, 9])
    assert (count, start, length, active) == (3, 2, 32, 0)


@pytest.mark.parametrize("fault", ["overrun", "negative", "zero_normal", "count_fraction", "nan"])
def test_invalid_active_contact_data_fail_closed(fault):
    data = list(buffers())
    if fault == "overrun": data[5] = np.array([[31]])
    if fault == "negative": data[0][2] = -1
    if fault == "zero_normal": data[2][2] = 0
    if fault == "count_fraction": data[4] = np.array([[2.5]])
    if fault == "nan": data[2][2, 0] = np.nan
    with pytest.raises(ValueError): mod.point_sum(data)


def test_empty_and_inactive_uninitialized_padding_ignored():
    data = list(buffers(force=()))
    data[0][:] = np.nan; data[2][:] = np.nan; data[5] = np.array([[-1]])
    assert np.array_equal(mod.point_sum(data)[0], [0, 0, 0])


def fixture():
    env = NS(num_envs=1, physics_dt=.0025, step_dt=.02, _sim_step_counter=0,
        cfg=NS(decimation=8, sim=NS(dt=.0025, gravity=(0., 0., -9.81))))
    env.sim = NS(current_time_step_index=0, current_time=0., get_physics_dt=lambda: .0025)
    robot = NS(body_names=["base", "left", "right"], root_physx_view=NS(get_masses=lambda: np.array([[20., 1., 1.]])))
    class Scene(dict): pass
    env.scene = Scene(robot=robot); env.scene.sensors = {}; env.scene.update = lambda dt: None
    for side, (old, new) in enumerate(zip(mod.LEGACY_SENSORS, mod.collector.SENSORS)):
        for name, cap in ((old, 32), (new, 64)):
            view = NS(filter_count=1, max_contact_data_count=cap,
                get_contact_force_matrix=lambda dt: np.array([[[0., 0., 9.]]]),
                get_contact_data=lambda dt: buffers())
            env.scene.sensors[name] = NS(body_names=[mod.collector.WHEELS[side]], contact_physx_view=view,
                cfg=NS(filter_prim_paths_expr=[mod.collector.GROUND], max_contact_data_count_per_prim=cap),
                data=NS(force_matrix_w=np.array([[[[0., 0., 9.]]]])))
    return env


def advance(env, n=1):
    env._sim_step_counter += n; env.sim.current_time_step_index += n
    env.sim.current_time += .0025*n; env.scene.update(.0025)


def raw():
    f = np.zeros((2793, 1, 2, 3)); f[..., 2] = 9
    return {"time_s": np.arange(2793)*.0025, "wheel_ground_contact_force_w_n": f,
            "wheel_ground_contact_count": np.zeros((2793, 1, 2), dtype=int)}


def test_actual_counter_only_observer_and_full_record_alignment():
    env = fixture(); step = object(); env.step = step
    probe = mod.ForceAPIObserver(env)
    env.scene.update(.0025); assert not probe.rows
    for _ in range(2792): advance(env)
    env.scene.update(.0025); assert len(probe.rows) == 2792
    arrays, proof = probe.finish(raw())
    assert env.step is step  # No envstep/reward owner.
    assert np.array_equal(arrays["counter"], np.arange(1, 2793))
    assert np.array_equal(arrays["time_s"], raw()["time_s"][1:])
    assert proof["comparisons"]["point_sum_vs_direct32"]["max_abs_vector_difference_n"] == 0
    assert proof["legacy32_possible_saturation"] is False
    assert proof["mass_and_gravity"]["mg_n_for_diagnostic_only"] == pytest.approx(22*9.81)


def test_missing_counter_or_partial_episode_rejected():
    env = fixture(); probe = mod.ForceAPIObserver(env)
    with pytest.raises(ValueError, match="all2792"): probe.finish(raw())
    with pytest.raises(ValueError, match="observer clock"): advance(env, 2)


def test_force_masks_compare_any_and_each_wheel_without_pointcount():
    a = np.zeros((2792, 2, 3)); b = a.copy()
    a[0, 0, 2] = 6; b[0, 1, 2] = 6; a[1, 0, 2] = 9
    result = mod.force_comparison(a, b)
    assert result["per_wheel_Fz_gt5N_mask_disagreement_samples"] == [2, 1]
    assert result["any_wheel_Fz_gt5N_mask_disagreement_samples"] == 1
    assert result["pointcount_gate"] is False


def test_probe_identity_distinct_and_default_native_cli_rejected_before_plan():
    assert mod.KIND != mod.base.KIND and mod.sha256(mod.__file__) != mod.BASE_SHA
    argv = sum((["--"+name, "fake"] for name in ("training-dir", "protocol", "legacy-evaluator", "motion-file", "output-dir")), [])
    with pytest.raises(ValueError, match="nominal full-horizon PhysX"): mod.prepare_plan(argv)


def test_immutable_A_sources_pinned_and_no_reward_install():
    assert mod.sha256(mod.base.__file__) == mod.BASE_SHA
    assert mod.sha256(mod.collector.__file__) == mod.COLLECTOR_SHA
    source = Path(mod.__file__).read_text()
    assert "collector.install(" not in source and "collector.reward_term(" not in source
    assert '"original_A_strict_loader_must_reject": True' in source


def test_original_A_strict_analyzer_rejects_probe_kind():
    import analyze_ppo_ground_contact_v2 as strict
    with pytest.raises(ValueError, match="provenance differs: kind"):
        strict._check_new_report({"ground_contact_v2_evaluation": {"kind": mod.KIND}}, {}, "physx",
            strict.DEFAULT_PROTOCOL, "f"*64, "a"*64)


def test_capacity_saturation_is_explicit_measurement_not_flight():
    env = fixture(); probe = mod.ForceAPIObserver(env)
    for name in mod.LEGACY_SENSORS:
        env.scene.sensors[name].contact_physx_view.get_contact_data = lambda dt: buffers(force=(1.,)*32, start=0)
    for _ in range(2792): advance(env)
    _, proof = probe.finish(raw())
    assert proof["legacy32_max_occupied_points_L_R"] == [32, 32]
    assert proof["legacy32_at_capacity_samples_L_R"] == [2792, 2792]
    assert proof["legacy32_possible_saturation"] is True
    assert "not proof" in proof["saturation_interpretation"]


def test_base_exclusive_audit_is_not_overwritten_by_probe(tmp_path):
    original = tmp_path/"evaluation_source_audit.json"
    mod.write_json(original, {"kind": mod.base.KIND})
    frozen = original.read_bytes()
    report = {"ground_contact_v2_evaluation": {"kind": mod.KIND, "role": mod.ROLE}}
    mod.write_probe_source_audit({"input_dir": tmp_path}, report)
    assert original.read_bytes() == frozen
    assert mod.base.read_json(tmp_path/"force_probe_source_audit.json") == report["ground_contact_v2_evaluation"]
    with pytest.raises(FileExistsError): mod.write_probe_source_audit({"input_dir": tmp_path}, report)


def test_r2_explicitly_preserves_failed_frozen_v1_source():
    old = ROOT/"scripts/eval_ppo_ground_force_api_probe_v2.py"
    assert mod.sha256(old) == "3c36b568592d9d3f4d242aa271f0e6d772365df15c8d09ac00e6c1e561761deb"
    source = Path(mod.__file__).read_text()
    assert 'write_json(plan["input_dir"]/"evaluation_source_audit.json"' not in source
    assert 'write_probe_source_audit(plan, report)' in source

