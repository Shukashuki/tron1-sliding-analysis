"""Native-buffer tests independent of Isaac and GPU availability."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_wheel_native_forces import reduce_native
from run_wheel_native_force_suite import gate


def buffers():
    n=(np.array([[999.],[10.],[20.],[999.]]),np.zeros((4,3)),np.tile([0.,0.,1.],(4,1)),
       np.zeros((4,1)),np.array([[2]]),np.array([[1]]))
    f=(np.array([[999.,999.,999.],[3.,-4.,0.],[999.,999.,999.],[999.,999.,999.]]),
       np.zeros((4,3)),np.array([[1]]),np.array([[1]]))
    return n,f


def test_native_offsets_exclude_stale_buffer_entries():
    r=reduce_native(*buffers())
    np.testing.assert_array_equal(r['normal_force_w_n'],[0,0,30])
    np.testing.assert_array_equal(r['friction_force_w_n'],[3,-4,0])
    assert r['normal_count']==2 and r['friction_count']==1


def test_empty_contact_does_not_read_arbitrary_start_offset():
    n,f=buffers()
    n=list(n); f=list(f); n[4]=f[2]=np.array([[0]]); n[5]=f[3]=np.array([[-99]])
    r=reduce_native(n,f)
    np.testing.assert_array_equal(r['normal_force_w_n'],[0,0,0])
    np.testing.assert_array_equal(r['friction_force_w_n'],[0,0,0])


def test_saturated_or_invalid_range_is_not_silently_truncated():
    n,f=buffers(); n=list(n); n[4]=np.array([[4]])
    with pytest.raises(ValueError): reduce_native(n,f)
    n[4]=np.array([[2]]); n[5]=np.array([[3]])
    with pytest.raises(ValueError): reduce_native(n,f)


def test_nonnormal_force_components_are_preserved():
    n,f=buffers(); n=list(n); n[2]=np.tile([.6,0.,.8],(4,1))
    r=reduce_native(n,f)
    np.testing.assert_allclose(r['normal_force_w_n'],[18,0,24])


def test_zero_utilization_with_existing_process_is_not_idle(monkeypatch):
    def output(args, **kwargs):
        if '--query-compute-apps=gpu_uuid,pid' in args: return 'GPU-B, 123\n'
        return '1, GPU-B, 00000000:02:00.0, 23000, 0\n'
    monkeypatch.setattr('run_wheel_native_force_suite.subprocess.check_output',output)
    with pytest.raises(RuntimeError,match='existing CUDA process'): gate(1)


def test_physical_index_is_mapped_by_pci_order(monkeypatch):
    def output(args, **kwargs):
        if '--query-compute-apps=gpu_uuid,pid' in args:return ''
        return '0, GPU-A, 00000000:02:00.0, 23000, 0\n1, GPU-B, 00000000:01:00.0, 23000, 0\n'
    monkeypatch.setattr('run_wheel_native_force_suite.subprocess.check_output',output)
    r=gate(1)
    assert r['physical_index']==1 and r['cuda_ordinal']==0 and r['pids']==[]
