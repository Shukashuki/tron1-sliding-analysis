"""N proof gates reject old M and smoke artifacts; criteria stay immutable."""
import hashlib
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_ppo_ground_contact_v2n as ppo
import analyze_cem_ground_contact_v2n as cem
import render_ppo_ground_contact_v2n as ppo_render
import render_cem_ground_contact_v2n as cem_render


def test_analysis_and_graphics_bind_actual_N_sources_and_immutable_metrics():
    ppo.load_protocol(); cem.load_protocol()
    assert ppo.recorded_metrics is ppo.original.recorded_metrics
    assert ppo.nominal_screen is ppo.original.nominal_screen
    assert ppo_render.ANALYZER_SHA256 == hashlib.sha256(Path(ppo.__file__).read_bytes()).hexdigest()
    assert cem_render.ANALYZER_SHA256 == hashlib.sha256(Path(cem.__file__).read_bytes()).hexdigest()
    assert ppo_render.GRAPHICS_SHA256 == cem_render.GRAPHICS_SHA256


def test_completed_M_PPO_cannot_be_presented_as_N():
    path = ROOT/"outputs/ground_contact_v2m_20261024_v1/ppo_training_proof_v1/ground_contact_v2m_training_audit.json"
    assert path.is_file(), "Historical completed M evidence required"
    with pytest.raises(ValueError, match="Copied N formal training audit differs"):
        ppo.load_training_audit(path)


def test_completed_M_CEM_cannot_be_presented_as_N():
    folder = ROOT/"outputs/ground_contact_v2m_20261024_v1/cem_actual_nominal_v1"
    assert (folder/"report.json").is_file()
    with pytest.raises(ValueError): cem.load_completed(folder)


def test_N_smoke_cannot_be_scored_as_completed_nominal_CEM():
    folder = ROOT/"outputs/ground_contact_v2n_20261025_v1/cem_actual_smoke_v1"
    assert (folder/"report.json").is_file()
    with pytest.raises(ValueError): cem.load_completed(folder)
