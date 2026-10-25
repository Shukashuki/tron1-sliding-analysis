"""Force auditing against real earlier native records; these are unit fixtures.

The fixture adapts only reset-first force semantics. It is never published as
the new actor's episode or used to claim a new model/controller outcome.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_ppo_k_nonlinear_tire as analysis
import collect_ppo_k_nonlinear_tire as collection


@pytest.fixture
def episode():
    folder=ROOT/'results/physx_tron1_wheel_models_20261025_v1/evidence/matched_models_v1/authored__combined_slip_magic_formula'
    tire=analysis.npz(folder/'tire_trajectory.npz')
    raw=analysis.npz(folder/'physics_trajectory.npz')
    reference=analysis.metrics.metrics.reference_data(analysis.metrics.DEFAULT_REFERENCE)
    _,trace=analysis.metrics.recorded_metrics(raw,'physx',reference)
    tire['history_valid']=np.ones(2792,dtype=bool);tire['history_valid'][0]=False
    tire['mu']=np.full((2792,1),.7,dtype=np.float32)
    for name in ('preceding_normal_load_n','applied_tangent_force_w_n','applied_tangent_moment_COM_w_nm'):
        tire[name][0]=0
    return tire,trace


def test_independent_cpu_law_accepts_recorded_forces(episode):
    result=analysis.audit_tire_trace(*episode)
    assert result['independent_NumPy_wrench_recomputation_passed']
    # The saved quaternion is float32; the auditor deliberately reconstructs
    # its axle in float64, unlike the original NumPy runtime's float32 frame.
    assert result['force_max_abs_difference_n']<1e-5
    assert result['moment_max_abs_difference_nm']<1e-5
    assert result['contact_energy_at_preceding_point_j']<0
    assert result['normal_load_delay_physical_steps']==1


@pytest.mark.parametrize('corruption',['moment_sign','load_delay','native_double_force','reset_force','loaded_nan_point'])
def test_algebra_coupling_and_double_counting_errors_rejected(episode,corruption):
    tire,trace=episode
    active=np.argwhere(tire['preceding_normal_load_n']>5)[0]
    i,j=active
    if corruption=='moment_sign':tire['applied_tangent_moment_COM_w_nm'][i,j]*=-1
    if corruption=='load_delay':tire['preceding_normal_load_n']=np.roll(tire['preceding_normal_load_n'],1,axis=0)
    if corruption=='native_double_force':tire['native_friction_force_w_n'][i,j,0]=1
    if corruption=='reset_force':tire['applied_tangent_force_w_n'][0,0,0]=1
    if corruption=='loaded_nan_point':tire['preceding_contact_position_w_m'][i,j]=np.nan
    with pytest.raises((AssertionError,ValueError)):analysis.audit_tire_trace(tire,trace)


def test_absence_of_any_wheel_load_is_an_analyzable_failure_not_missing_data(episode):
    tire,trace=episode
    trace['floor_normal_force'][:]=0
    for key in ('preceding_normal_load_n','applied_tangent_force_w_n','applied_tangent_moment_COM_w_nm'):
        tire[key][:]=0
    result=analysis.audit_tire_trace(tire,trace)
    assert result['core_loaded_wheel_samples']==0
    assert result['core_used_load_utilization_at_least0p9_fraction'] is None
    assert result['contact_energy_at_preceding_point_j']==0


def test_incomplete_native_comparison_cannot_be_collected(tmp_path):
    (tmp_path/'nominal_v1').mkdir()
    (tmp_path/'driver_status.json').write_text(json.dumps({'status':'running_formal'}))
    (tmp_path/'nominal_v1/status.json').write_text(json.dumps({'status':'waiting_for_exact_training_driver'}))
    with pytest.raises(AssertionError):collection.verify_completed(tmp_path)
    assert not list(tmp_path.glob('*.tar.gz'))


def test_substituted_reference_is_rejected_before_scoring(tmp_path):
    from types import SimpleNamespace
    fake=tmp_path/'other_reference.npz';np.savez(fake,position=np.zeros((2,3)))
    with pytest.raises(AssertionError,match='exact training reference'):
        analysis.run(SimpleNamespace(output_dir=tmp_path/'analysis',reference=fake,collection=tmp_path))
