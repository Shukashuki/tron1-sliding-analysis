"""Fake guard/children only; no native simulation or actor is executed."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_ppo_contact_numerics as runner
import eval_ppo_contact_numerics as evaluator
from test_eval_ppo_contact_numerics import make_plan, fixture_env, readback, BASE_REPORT, LEGACY
from run_ppo_smoothness_sequence import write_json


@pytest.fixture
def args(tmp_path, monkeypatch):
    inputs = tmp_path/"inputs"; inputs.mkdir()
    values = {name: inputs/name for name in ("checkpoint", "guard_source", "motion_file", "asset_path", "runner_config", "task_profile")}
    for path in values.values(): path.write_text("CPU fixture")
    bm = inputs/"bm"; config = bm/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    config.parent.mkdir(parents=True); config.write_text("# CPU config fixture\n")
    original = runner.base.sha256
    monkeypatch.setattr(runner.base, "sha256", lambda path: runner.base.CHECKPOINT_SHA if Path(path) == values["checkpoint"] else original(path))
    monkeypatch.setattr(runner.base, "freeze", lambda args: {"frozen": 1})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return NS(**values, legacy_evaluator=LEGACY, beyondmimic_source=bm, python=Path(sys.executable), output_dir=tmp_path/"fresh")


def make_output(folder, iterations, monkeypatch):
    folder.mkdir()
    plan = make_plan(iterations, monkeypatch); cfg, env, _ = fixture_env(iterations); readback(cfg, env, plan)
    report = json.loads(BASE_REPORT.read_text()); report.pop("contact_stability_diagnostic")
    report["contact_numerics_diagnostic"] = plan["audit"]
    report["evaluator_source_sha256"] = evaluator.base.sha256(evaluator.__file__)
    report["policy_export"].pop("contact_stability_diagnostic")
    evaluator.decorate_contract(report["policy_export"], plan)
    (folder/"physics_trajectory.npz").symlink_to(BASE_REPORT.parent/"physics_trajectory.npz")
    write_json(folder/"report.json", report)


class Guard:
    KIT_ARGS = "--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false --/physics/cudaDevice=1 --/physics/sceneMultiGPUMode=0"
    write_json = staticmethod(write_json)
    def __init__(self, args, monkeypatch, *, busy_at=None, nonzero=False):
        self.args, self.monkeypatch = args, monkeypatch; self.probes = self.calls = 0; self.busy_at, self.nonzero = busy_at, nonzero
    def query_gpu_one(self):
        self.probes += 1; return {"physical_index": 1, "idle": self.probes != self.busy_at}
    def run_child(self, command, log, cwd, environment, spawned):
        self.calls += 1; spawned(7000+self.calls); log.write_text("CPU fixture\n")
        assert "CUDA_VISIBLE_DEVICES" not in environment and environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert environment["TRON1_BEYONDMIMIC_ROOT"] == str(self.args.beyondmimic_source)
        assert "--leg-kp" not in command and "--leg-kd" not in command
        assert command[command.index("--kit_args")+1] == self.KIT_ARGS
        iterations = int(command[command.index("--solver-velocity-iterations")+1])
        make_output(Path(command[command.index("--output-dir")+1]), iterations, self.monkeypatch)
        return 1 if self.nonzero else 0


def test_full_sequence_two_fresh_idle_checks_baseline_reproduced(args, monkeypatch):
    guard = Guard(args, monkeypatch); sleeps = []
    status = runner.run(args, guard=guard, quiesce=sleeps.append)
    assert status["status"] == "episodes_complete_requires_contact_and_tracking_review"
    assert guard.calls == guard.probes == 2 and sleeps == [2., 2.]
    assert status["children"][0]["verification"]["recording_sha256"] == runner.BASELINE_RAW_SHA256
    assert status["paired_native_offsets_and_materials_identical"] is True
    assert status["native_solver_count_claimed"] is status["contact_success_claimed"] is False
    assert all(row["status"] == "completed_verified" for row in status["children"])


@pytest.mark.parametrize("busy_at", [1, 2])
def test_gpu_busy_never_launches_following_child(args, monkeypatch, busy_at):
    guard = Guard(args, monkeypatch, busy_at=busy_at)
    with pytest.raises(ValueError, match="fully idle"):
        runner.run(args, guard=guard, quiesce=lambda s: None)
    assert guard.probes == busy_at and guard.calls == busy_at-1
    status = json.loads((args.output_dir/"contact_numerics_status.json").read_text())
    assert status["status"] == "stopped_without_retry" and status["last_gpu_preflight"]["idle"] is False


def test_failed_own_child_not_retried(args, monkeypatch):
    guard = Guard(args, monkeypatch, nonzero=True)
    with pytest.raises(ValueError, match="Owned evaluator failed"):
        runner.run(args, guard=guard, quiesce=lambda s: None)
    assert guard.calls == 1
    assert json.loads((args.output_dir/"contact_numerics_status.json").read_text())["children"][0]["status"] == "failed_or_unverified"


def test_solver4_must_match_original_before8(args, monkeypatch):
    guard = Guard(args, monkeypatch); monkeypatch.setattr(runner, "BASELINE_RAW_SHA256", "0"*64)
    with pytest.raises(ValueError, match="reproduce original baseline"):
        runner.run(args, guard=guard, quiesce=lambda s: None)
    assert guard.calls == 1


def test_paired_offset_or_material_changes_are_not_single_factor(args, monkeypatch):
    first = args.output_dir.parent/"a"; second = args.output_dir.parent/"b"
    make_output(first, 4, monkeypatch); make_output(second, 8, monkeypatch)
    before, after = runner.validate_output(first, 4), runner.validate_output(second, 8)
    runner.compare_readbacks(before, after)
    after["runtime_readback"]["native_shape_readback"]["rest_offsets_m"][0][0] += .001
    with pytest.raises(ValueError, match="Additional native"):
        runner.compare_readbacks(before, after)


def test_existing_output_or_missing_official_source_prevents_launch(args):
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="Preserve existing"):
        runner.run(args)
