"""O offline gates retain original metrics and reject N/smoke substitutions."""
import hashlib
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_ppo_ground_contact_v2o as ppo
import analyze_cem_ground_contact_v2o as cem
import render_ppo_ground_contact_v2o as ppo_render
import render_cem_ground_contact_v2o as cem_render


def test_all_frozen_protocol_source_pins_metrics_and_graphics_bindings():
    ppo.load_protocol(); cem.load_protocol()
    assert ppo.recorded_metrics is ppo.original.recorded_metrics
    assert ppo.nominal_screen is ppo.original.nominal_screen
    assert ppo_render.ANALYZER_SHA256 == hashlib.sha256(Path(ppo.__file__).read_bytes()).hexdigest()
    assert cem_render.ANALYZER_SHA256 == hashlib.sha256(Path(cem.__file__).read_bytes()).hexdigest()
    assert ppo_render.GRAPHICS_SHA256 == cem_render.GRAPHICS_SHA256
    assert "ppo_ground_contact_v2k.py" in ppo.TRAINING_SOURCE_PINS
    assert "scripts/nmpc_batch_ground_contact_v2k_400hz.py" in cem.SOURCE_PINS
    assert len(cem.SOURCE_PINS) == 26


def test_completed_N_PPO_is_not_O_training():
    path = ROOT/"outputs/ground_contact_v2n_20261025_v1/ppo_training_proof_v1/ground_contact_v2n_training_audit.json"
    assert path.is_file()
    with pytest.raises(ValueError): ppo.load_training_audit(path)


@pytest.mark.parametrize("folder", ["ground_contact_v2n_20261025_v1/cem_actual_nominal_v1",
                                  "ground_contact_v2o_20261025_v1/cem_actual_smoke_v1"])
def test_old_N_or_O_smoke_cannot_be_scored_as_actual_O_nominal(folder):
    path = ROOT/"outputs"/folder
    assert (path/"report.json").is_file()
    with pytest.raises(ValueError): cem.load_completed(path)
