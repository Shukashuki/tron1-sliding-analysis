"""Opt-in dispatch/lifetime regressions without creating an Isaac simulation."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
pytest.importorskip("mujoco")

import run_nmpc_tracking as mujoco_runner
from run_nmpc_tracking_isaac import (
    add_prediction_backend_arguments, prediction_backend_options, build_prediction_backend,
)


def test_isaac_parser_physx_selection_is_explicit_and_requires_existing_config(tmp_path):
    parser = add_prediction_backend_arguments(argparse.ArgumentParser())
    assert prediction_backend_options(parser.parse_args([])) == ("python", None)
    config = tmp_path/"physx.json"
    with pytest.raises(ValueError, match="existing config"):
        prediction_backend_options(parser.parse_args([
            "--prediction-backend", "physx", "--physx-prediction-config", str(config)]))
    config.write_text('{"device":"cuda:3"}')
    args = parser.parse_args(["--prediction-backend", "physx", "--physx-prediction-config", str(config)])
    assert prediction_backend_options(args) == ("physx", None)
    with pytest.raises(ValueError, match="existing config"):
        prediction_backend_options(NS(prediction_backend="physx", physx_prediction_config=config,
                                      native_library=tmp_path/"legacy.so"))
    with pytest.raises(ValueError, match="only valid"):
        prediction_backend_options(NS(prediction_backend="python", physx_prediction_config=config))


def test_isaac_build_passes_actual_objective_and_config_to_physx_adapter(monkeypatch, tmp_path):
    calls = []
    class FakeAdapter:
        def __init__(self, prediction, config):
            calls.append((prediction, prediction.objective, config))
    monkeypatch.setitem(sys.modules, "physx_batch_predictor", NS(PhysXBatchTrackingPredictor=FakeAdapter))
    config = tmp_path/"physx.json"
    payload = {"schema_version": 1, "device": "cuda:3", "environment_spacing_m": 0.}
    config.write_text(json.dumps(payload))
    limits = np.tile([-1., 1.], (8, 1))
    prediction = NS(objective=NS(soft_joint_limits=limits))
    assert isinstance(build_prediction_backend(prediction, "physx", None, physx_config=config), FakeAdapter)
    assert calls == [(prediction, prediction.objective, payload)]
    assert calls[0][1].soft_joint_limits is limits
    with pytest.raises(ValueError, match="explicit configuration"):
        build_prediction_backend(prediction, "physx", tmp_path/"native.so", physx_config=config)


def test_default_isaac_backend_never_constructs_physx_or_native_worker(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Default Python path started a worker")
    monkeypatch.setitem(sys.modules, "physx_batch_predictor", NS(PhysXBatchTrackingPredictor=forbidden))
    monkeypatch.setitem(sys.modules, "nmpc_batch_objective", NS(NativeBatchTrackingPredictor=forbidden))
    prediction = object()
    assert build_prediction_backend(prediction, "python", None) is prediction


class BeforePlant(RuntimeError):
    """End a runner integration check before any plant is constructed."""


@pytest.fixture
def minimal_runner(monkeypatch, tmp_path):
    profile, contract, config = (tmp_path/name for name in ("task.json", "contract.json", "physx.json"))
    profile.write_text("{}")
    contract.write_text("{}")
    config.write_text('{"schema_version":1,"device":"cuda:3","max_candidates":64}')
    args = mujoco_runner.parser().parse_args([
        "--model", str(tmp_path/"model.xml"), "--motion", str(tmp_path/"motion.npz"),
        "--contract", str(contract), "--task-profile", str(profile),
        "--plant-mu", "0.7", "--output-dir", str(tmp_path/"not_created")])
    reference = NS(frames=2, fps=50.)
    def before_plant(*args, **kwargs):
        raise BeforePlant("controller constructed; plant was not created")
    provider = NS(reference=reference, api=NS(load_model=before_plant))
    scalar = NS(provider=provider, reference=reference, objective=object())
    monkeypatch.setattr(mujoco_runner, "selected_reference", lambda _: {"frames": 2, "fps": 50.})
    monkeypatch.setattr(mujoco_runner, "effective_settings", lambda _: ({}, NS(candidates=64), (.7,)))
    monkeypatch.setattr(mujoco_runner, "build_provider", lambda *a: provider)
    monkeypatch.setattr(mujoco_runner, "MujocoTrackingPredictor", lambda *a: scalar)
    monkeypatch.setattr(mujoco_runner, "attach_model_nominal_plan", lambda *a: {"enabled": False})
    monkeypatch.setattr(mujoco_runner, "ShootingNMPC", lambda prediction, cfg, **kw: NS(predictor=prediction, config=cfg))
    return NS(args=args, config=config, scalar=scalar, output=Path(args.output_dir))


def fake_physx_module(monkeypatch):
    instances = []
    class FakePhysX:
        def __init__(self, base, config):
            self.base, self.config = base, config
            self.metadata = {"engine": "Isaac/PhysX"}
            self.closed, self.wired, self.entered = False, False, False
            instances.append(self)
        def __enter__(self):
            self.entered = True
            return self
        def __exit__(self, *args):
            self.closed = True
        def record_cem_wiring(self, controller):
            assert controller.predictor is self
            self.wired = True
    monkeypatch.setitem(sys.modules, "physx_batch_predictor", NS(PhysXBatchTrackingPredictor=FakePhysX))
    return instances


def test_mujoco_runner_optin_creates_one_owned_physx_adapter(monkeypatch, minimal_runner):
    fixture = minimal_runner
    instances = fake_physx_module(monkeypatch)
    fixture.args.prediction_backend = "physx"
    fixture.args.physx_prediction_config = fixture.config
    with pytest.raises(BeforePlant):
        mujoco_runner.run(fixture.args)
    assert len(instances) == 1
    adapter = instances[0]
    assert adapter.base is fixture.scalar
    assert adapter.config["device"] == "cuda:3"
    assert adapter.entered and adapter.wired and adapter.closed
    assert not fixture.output.exists()


def test_mujoco_runner_closes_physx_on_controller_construction_failure(monkeypatch, minimal_runner):
    fixture = minimal_runner
    instances = fake_physx_module(monkeypatch)
    fixture.args.prediction_backend = "physx"
    fixture.args.physx_prediction_config = fixture.config
    def fail_controller(*args, **kwargs):
        raise BeforePlant("controller creation failed")
    monkeypatch.setattr(mujoco_runner, "ShootingNMPC", fail_controller)
    with pytest.raises(BeforePlant, match="creation failed"):
        mujoco_runner.run(fixture.args)
    assert len(instances) == 1 and instances[0].entered and instances[0].closed
    assert not instances[0].wired


def test_default_mujoco_runner_never_starts_physx_worker(monkeypatch, minimal_runner):
    instances = fake_physx_module(monkeypatch)
    with pytest.raises(BeforePlant):
        mujoco_runner.run(minimal_runner.args)
    assert instances == []


@pytest.mark.parametrize("backend,has_config,has_library", [
    ("physx", False, False), ("python", True, False), ("native", True, True),
    ("physx", True, True),
])
def test_mujoco_runner_rejects_incompatible_optin_flags_before_worker(
        monkeypatch, minimal_runner, backend, has_config, has_library):
    fixture = minimal_runner
    instances = fake_physx_module(monkeypatch)
    fixture.args.prediction_backend = backend
    fixture.args.physx_prediction_config = fixture.config if has_config else None
    fixture.args.native_library = Path("prebuilt.so") if has_library else None
    with pytest.raises(ValueError):
        mujoco_runner.run(fixture.args)
    assert instances == []
    assert not fixture.output.exists()
