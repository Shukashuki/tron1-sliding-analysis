"""Validate immutable GPU diagnostic artifacts, not a new GPU benchmark."""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/"results/mjlab_gpu_task_20261021_v1"


def read_report(name):
    return json.loads((EVIDENCE/name).read_text())


@pytest.mark.parametrize("version,expected", [
    (9, "87ffed64ab9a04386a690d2eca4db857300584007ac169c037ef488621e71d88"),
    (10, "5429b5056a3588c7b0c81c2bbd8bcb2f5f649e5e45d9cc03451b61a9d8db00e7"),
])
def test_executed_source_archives_and_each_manifest_member_are_unchanged(version, expected):
    archive = EVIDENCE/f"mjlab_gpu_task_source_v{version}.tar.gz"
    manifest = read_report(f"mjlab_gpu_task_source_v{version}.tar.manifest.json")
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
    with tarfile.open(archive, "r:gz") as bundle:
        assert json.load(bundle.extractfile("snapshot_manifest.json")) == manifest
        assert set(bundle.getnames()) == {entry["path"] for entry in manifest["files"]} | {"snapshot_manifest.json"}
        for entry in manifest["files"]:
            payload = bundle.extractfile(entry["path"]).read()
            assert len(payload) == entry["size"]
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("report_name,version", [("benchmark_report.json", 9), ("profile_report.json", 10)])
def test_report_source_hashes_resolve_to_exact_executed_snapshot(report_name, version):
    report = read_report(report_name)
    manifest = read_report(f"mjlab_gpu_task_source_v{version}.tar.manifest.json")
    sources = {Path(entry["path"]).name: entry["sha256"] for entry in manifest["files"]}
    for name, value in report["source_sha256"].items():
        assert sources[name] == value
    assert report["training_speedup_claimed"] is False


def test_published_timing_medians_and_scope_are_supported_by_samples():
    report = read_report("benchmark_report.json")
    assert report["status"] == "passed_parity_and_measured_environment_timing"
    assert report["parity"]["passed"]
    assert report["parity"]["uninterrupted_no_auto_resets"]
    assert report["timing_task_trace_parity_claimed"] is False
    for metrics in report["environment_timing"].values():
        median = statistics.median(metrics["seconds_per_window"])
        assert metrics["median_window_s"] == median
        assert metrics["transitions_per_window"] == 512*24
        assert metrics["median_transitions_per_s"] == pytest.approx(512*24/median)
    assert report["measured_environment_step_speedup"] == pytest.approx(
        report["environment_timing"]["host"]["median_window_s"]/
        report["environment_timing"]["gpu"]["median_window_s"])


def test_profile_is_marked_non_additive_and_not_an_optimization_result():
    report = read_report("profile_report.json")
    assert report["optimization_applied"] is False
    assert report["physics_or_task_parameters_changed"] is False
    assert report["training_run"] is False
    profile = report["profile"]
    assert profile["stage_totals_are_additive"] is False
    assert profile["stages"]["physics.step"]["calls"] == 8*4
    assert profile["stages"]["physics.forward"]["calls"] == 8*4
    assert all(stage["inclusive_non_additive"] for stage in profile["stages"].values())
