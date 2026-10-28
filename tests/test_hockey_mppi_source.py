"""Source admission checks only; synthetic reports are not simulated evidence."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_hockey_physx import sha,verify_hockey_search_source


def fixture():
    root=Path(__file__).resolve().parents[1]
    source=root/'results/hockey_stop_mppi_preflight_20261027_v1/source_snapshot_v1'
    cfg=json.loads((source/'config.json').read_text())
    files={'source_sha256':'generate_hockey_physx_mppi.py','task_math_sha256':'hockey_mppi_math.py',
           'config_sha256':'config.json','motion_sha256':'motion.npz',
           'conditioning_table_sha256':'nominal_conditioning.npz','initial_mean_sha256':'initial_mean.npz',
           'math_source_sha256':'physx_reference_math.py'}
    run={k:sha(source/v) for k,v in files.items()}
    run.update(kind='hockey_native_physx_mppi_v1',conditioning_mode='reference_dq',actual_actor_input_archived=True,
               policy_weights_changed=False,policy_training=False,external_forces=0,single_world_validation=True,
               worlds=1,iterations=0,search_history=[],actor_sha256=cfg['actor_sha256'],asset_sha256=cfg['asset_sha256'],
               tracking_source_sha256=cfg['tracking_sha256'])
    return source,run


def test_reviewed_source_is_admitted_without_implying_stop_success():
    s,r=fixture();result=verify_hockey_search_source(s,r)
    assert result['planning_costs_used_for_success'] is False
    assert result['qualification_thresholds_unchanged'] is True


@pytest.mark.parametrize('change',[{'worlds':64},{'single_world_validation':False},{'policy_weights_changed':True},
                                  {'external_forces':1},{'iterations':6},{'motion_sha256':'0'*64}])
def test_search_population_changed_actor_or_changed_target_cannot_be_final_validation(change):
    s,r=fixture();r.update(change)
    with pytest.raises(ValueError):verify_hockey_search_source(s,r)
