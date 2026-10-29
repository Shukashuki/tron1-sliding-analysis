"""Inactive contexts are allowed only for the exact owned job on another GPU."""
import copy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dispatch_hockey_bilateral_capacity import context_eligibility


def fixture():
    argv=['python','/data/zengxiong/workspaces/tron1-hockey-speed-success-20261028-v1/source_snapshot_v1/train_hockey_ppo.py','--device','cuda:3']
    process=dict(uid=1030,argv=argv,start_ticks=100)
    job=dict(argv=argv,identity=dict(start_ticks=100))
    gpu=dict(physical_index=1,utilization_percent=0,memory_used_mib=358,
             all_reported_compute_and_graphics_processes=[dict(pid='42',used_memory='346 MiB')])
    return gpu,{42:job},process


def test_small_owned_context_on_other_device_is_eligible():
    g,j,p=fixture();assert context_eligibility(g,j,lambda pid:p,1030)[0]


def test_zero_utilization_does_not_authorize_active_device_or_foreign_process():
    for change in ('device','owner','pid','identity','memory','utilization'):
        g,j,p=fixture();p=copy.deepcopy(p)
        if change=='device':g['physical_index']=3
        if change=='owner':p['uid']=1029
        if change=='pid':g['all_reported_compute_and_graphics_processes'][0]['pid']='43'
        if change=='identity':p['start_ticks']=101
        if change=='memory':g['all_reported_compute_and_graphics_processes'][0]['used_memory']='800 MiB'
        if change=='utilization':g['utilization_percent']=1
        assert not context_eligibility(g,j,lambda pid:p,1030)[0],change


def test_empty_device_is_eligible_without_trusted_jobs():
    g,_,_=fixture();g['all_reported_compute_and_graphics_processes']=[];g['memory_used_mib']=6
    assert context_eligibility(g,{},lambda pid:None,1030)[0]
