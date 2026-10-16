from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_controller_reference as prepare


def test_source_blob_hash_verifier_rejects_modified_csv_without_loading_assets(tmp_path):
    path = tmp_path / "source.csv"
    path.write_bytes(b"pinned fixture CSV")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert prepare.verify_file_hash(path, expected, "source CSV") == expected
    path.write_bytes(b"tampered fixture CSV")
    with pytest.raises(ValueError, match="source CSV SHA-256 mismatch"):
        prepare.verify_file_hash(path, expected, "source CSV")


def test_existing_output_is_preserved_before_loading_any_source_or_asset(tmp_path, monkeypatch):
    output = tmp_path / "previous"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"keep")
    monkeypatch.setattr(prepare, "verify_source_inputs", lambda *a, **kw: pytest.fail("should not load inputs"))
    with pytest.raises(FileExistsError):
        prepare.prepare_reference(model="missing.xml", urdf="missing.urdf", output_dir=output)
    assert sentinel.read_bytes() == b"keep"


def test_wrong_asset_hash_fails_before_creating_output(tmp_path, monkeypatch):
    model, urdf = tmp_path / "wrong.xml", tmp_path / "wrong.urdf"
    model.write_bytes(b"not the licensed export model")
    urdf.write_bytes(b"not the original URDF")
    monkeypatch.setattr(prepare, "verify_source_inputs", lambda *a, **kw: {"spec": prepare.get_reference()})
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="licensed export MJCF SHA-256 mismatch"):
        prepare.prepare_reference(model=model, urdf=urdf, output_dir=output)
    assert not output.exists()


def test_source_module_binding_restores_previous_process_modules(tmp_path, monkeypatch):
    module_names = ("export_tracking_motion", "capture_car_drift",
                    "prepare_car_drift_reference", "prepare_drift_drl_turn_reference")
    pins = {}
    for name in module_names:
        relative = f"scripts/{name}.py"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        text = "VALUE = 42\n" if name == module_names[0] else "from export_tracking_motion import VALUE\n"
        path.write_text(text)
        pins[relative] = prepare.sha256(path)
    prior = SimpleNamespace(VALUE=999)
    monkeypatch.setitem(sys.modules, module_names[0], prior)
    before = {name: sys.modules.get(name) for name in module_names}
    monkeypatch.setattr(prepare, "PINNED_SOURCE_FILES", pins)
    with prepare._source_modules(tmp_path) as modules:
        assert modules.export.VALUE == modules.turn.VALUE == 42
        assert sys.modules[module_names[0]] is not prior
    for name in module_names:
        assert sys.modules.get(name) is before[name]


@pytest.mark.parametrize("reference_id", [prepare.PRIMARY_REFERENCE_ID, prepare.SUPPLEMENT_REFERENCE_ID])
def test_adaptation_parameters_are_explicit_historical_values(reference_id):
    spec = prepare.get_reference(reference_id)
    config = {"spatial_scale": .08, "time_scale": 2., "root_height_m": .75,
              "recovery_s": 1.5, "stationary_tail_s": .5}
    assert prepare.adaptation_options(spec, config) == {
        "spatial_scale": .08, "time_scale": 2., "root_height": .75,
        "recovery_s": 1.5, "stationary_tail_s": .5, "output_fps": 50.}
    if reference_id == prepare.PRIMARY_REFERENCE_ID:
        config["time_scale"] = 1.6
        with pytest.raises(ValueError, match="timing/scaling"):
            prepare.adaptation_options(spec, config)


@pytest.mark.parametrize("reference_id", [prepare.PRIMARY_REFERENCE_ID, prepare.SUPPLEMENT_REFERENCE_ID])
def test_wrapper_uses_explicit_assets_offline_and_preserves_phase_masks(tmp_path, monkeypatch, reference_id):
    spec = prepare.get_reference(reference_id)
    model, urdf = tmp_path / "licensed.xml", tmp_path / "original.urdf"
    model.write_bytes(b"fixture model not a robot")
    urdf.write_bytes(b"fixture urdf not a robot")
    spec["motion_contract"]["export_model_sha256"] = prepare.sha256(model)
    spec["motion_contract"]["export_urdf_sha256"] = prepare.sha256(urdf)
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    source_root = tmp_path / "source"
    (source_root / "scripts").mkdir(parents=True)
    (source_root / "scripts/capture_car_drift.py").write_text("# fixture")
    inputs = {"spec": spec, "source_root": source_root, "head": spec["source"]["commit"],
              "payload": b"fixture raw CSV", "csv_file": Path(spec["source"]["csv_path"]),
              "license_payload": b"MIT fixture license", "configuration_file": config_file,
              "configuration": {"spatial_scale": .08, "time_scale": 2., "root_height_m": .75,
                                "recovery_s": 1.5, "stationary_tail_s": .5,
                                "source_row_start_inclusive": 1535}}
    monkeypatch.setattr(prepare, "verify_source_inputs", lambda *a, **kw: inputs)
    n = spec["frames"]
    phase = np.repeat([0, 1, 2], [spec["source_demo_frames"], 75, 25])
    calls = {}

    def adapt(source, **options):
        calls["adapt_options"] = options
        return {"source_row_numbers": np.zeros(n), "metadata": {"declared_fixture": True}}

    def solve(intent, *, model_path):
        calls["solve_model"] = model_path
        calls["source_row_numbers"] = intent["source_row_numbers"].copy()
        return {"time_s": np.arange(n) / 50., "demonstration_time_s_estimated": np.arange(n) / 100.,
                "source_phase": phase.copy(), "root_yaw_rad": np.zeros(n)}, {"simulation_steps": 0}

    def convert(robot_path, **kwargs):
        calls["convert"] = kwargs
        assert Path(robot_path).is_file()
        return {"time_s": np.arange(n) / 50., "joint_tracking_mask": np.array([True] * 6 + [False] * 2)}, {"warnings": []}

    def verify_capture(path):
        path = Path(path)
        assert path.parent.name == "sa_traj29"
        audit = json.loads((path.parent.parent / "source_audit.json").read_text())
        assert audit["source_commit"] == inputs["head"]
        assert audit["captures"]["sa_traj29"]["source_sha256"] == spec["source"]["csv_sha256"]
        assert (path.parent.parent / "raw" / inputs["csv_file"].name).read_bytes() == inputs["payload"]
        calls["offline_capture_verified"] = True
        return {}, {"source_id": "sa_traj29"}

    @contextmanager
    def modules(_):
        yield SimpleNamespace(
            car=SimpleNamespace(adapt_source=adapt, solve_reference=solve, load_verified_source=verify_capture),
            turn=SimpleNamespace(decode_csv=lambda payload: {}, selected_source=lambda full, cfg: ({}, {"fixture": True}),
                                 reference_turn_evidence=lambda arrays, cfg: {"fixture_turn": True}),
            capture=SimpleNamespace(decode_csv=lambda payload: {"fixture": np.arange(3)}),
            export=SimpleNamespace(convert_motion=convert))

    monkeypatch.setattr(prepare, "_source_modules", modules)
    output = tmp_path / "new"
    report = prepare.prepare_reference(reference_id=reference_id, model=model, urdf=urdf, output_dir=output)
    assert calls["solve_model"] == model.resolve()
    assert calls["convert"] == {"model_path": model.resolve(), "urdf_path": urdf.resolve(),
                                "output_fps": 50., "prepend_s": 0., "append_s": 0., "z_offset": 0.}
    assert calls["adapt_options"]["root_height"] == .75
    assert report["status"] == "hash_mismatch"  # Fixtures must never replace real locks.
    assert report["catalog_lock_matched"] is False
    assert report["expected_hashes"]["motion_npz_sha256"] == spec["motion_npz_sha256"]
    assert report["network_requests"] == report["simulation_steps"] == 0
    assert report["model_sha256"] == prepare.sha256(model)
    assert json.loads((output / "reproduction_report.json").read_text())["status"] == "hash_mismatch"
    with np.load(output / "motion.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["source_phase"], phase)
        np.testing.assert_array_equal(archive["joint_tracking_mask"], [True] * 6 + [False] * 2)
        assert archive["demonstration_time_s_estimated"].shape == (n,)
        assert archive["root_yaw_rad"].shape == (n,)
    if reference_id == prepare.PRIMARY_REFERENCE_ID:
        assert np.all(calls["source_row_numbers"] == 1535)
    else:
        assert calls["offline_capture_verified"] is True
        assert np.all(calls["source_row_numbers"] == 0)


def test_cli_requires_assets_and_returns_nonzero_on_hash_mismatch(tmp_path, monkeypatch):
    calls = {}

    def fake_prepare(**kwargs):
        calls.update(kwargs)
        return {"catalog_lock_matched": False, "status": "hash_mismatch"}

    monkeypatch.setattr(prepare, "prepare_reference", fake_prepare)
    with pytest.raises(SystemExit):
        prepare.main(["--output-dir", str(tmp_path / "missing_assets")])
    result = prepare.main(["--model", "asset.xml", "--urdf", "asset.urdf",
                           "--output-dir", str(tmp_path / "new")])
    assert result == 2
    assert calls["reference_id"] == prepare.PRIMARY_REFERENCE_ID
    assert calls["model"] == Path("asset.xml")
    assert calls["urdf"] == Path("asset.urdf")
