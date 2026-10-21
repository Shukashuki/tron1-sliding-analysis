"""Replay published GPU smoke evidence without simulator assets or CUDA."""
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"results/ppo_mjlab_backend_preflight_20261021_v1"


def read(name):
    return json.loads((DATA/name).read_text())


def test_real_gpu_runtime_and_contact_preflight():
    runtime = read("runtime_preflight.json")
    assert runtime["passed"] and runtime["warp_kernel_passed"]
    assert runtime["mjlab_commit"] == "4360096b49c1bcc478668c11ce7d458db747e49d"
    assert runtime["simulation"]["physics_steps"] == 64
    assert runtime["simulation"]["max_contacts"] > 0
    assert not runtime["simulation"]["cuda_graphs_enabled"]


def test_actual_tron1_arrays_have_real_contacts_and_isolated_reset():
    report = read("tron1_physics_smoke.json")
    assert report["status"] == "passed_smoke_not_full_training"
    assert report["physics_metadata"]["host_task_adapter"]
    with np.load(DATA/"tron1_physics_smoke_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["initial_actor"].shape == (4, 51)
        assert arrays["initial_critic"].shape == (4, 120)
        assert arrays["contact_force_history_w"].shape == (4, 4, 9, 3)
        assert report["actual_contact_force_norm_N"] > 0
        np.testing.assert_allclose(np.linalg.norm(arrays["contact_force_history_w"]),
                                   report["actual_contact_force_norm_N"])
        np.testing.assert_allclose(arrays["after_subset_reset_time"][0], 0.)
        np.testing.assert_array_equal(arrays["after_subset_reset_time"][1:],
                                      arrays["before_subset_reset_time"][1:])


def test_both_smokes_complete_with_identical_initial_policy_not_performance_claims():
    states = [read(name+"_training_smoke_status.json") for name in ("mjlab", "physx")]
    assert len({s["initial_policy_state_sha256"] for s in states}) == 1
    for state in states:
        assert state["status"] == "training_completed"
        assert state["training_environment_steps"] == 32*24*2
        assert state["performance_claimed"] is False
    physx = read("physx_training_smoke_manifest.json")
    assert len(physx["environment"]["reward_terms"]) == 15
    assert physx["environment"]["termination_terms"] == ["time_out", "anchor_ori", "motion_end"]
    assert physx["environment"]["ground_material_usd_verified"]


def test_exact_executed_snapshots_contain_only_manifested_source():
    for stem in ("executed_mjlab_smoke_snapshot_v3", "executed_physx_smoke_snapshot_v6"):
        manifest = read("source/"+stem+".manifest.json")
        with tarfile.open(DATA/"source"/(stem+".tar.gz"), "r:gz") as archive:
            expected = {item["path"]: item for item in manifest["files"]}
            assert set(archive.getnames()) == set(expected)|{"snapshot_manifest.json"}
            assert json.loads(archive.extractfile("snapshot_manifest.json").read()) == manifest
            for name, entry in expected.items():
                blob = archive.extractfile(name).read()
                assert len(blob) == entry["size"]
                assert hashlib.sha256(blob).hexdigest() == entry["sha256"]
                assert name.endswith((".py", ".cpp", ".json", "LICENSE", "LICENSE.md", "LICENCE"))
