"""Check volume, envelope and parallel-axis inertia invariance before PhysX."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from wheel_contact_geometry import UniformWheel


@pytest.mark.parametrize("count", [1, 3, 5, 9])
def test_partition_preserves_envelope_volume_COM_and_inertia(count):
    wheel = UniformWheel(slices=count)
    pieces = wheel.segments()
    mass, first_moment, inertia, volume = 0., 0., np.zeros(3), 0.
    previous_edge = -wheel.width_m/2
    for p in pieces:
        y, h, r = p["center_y_m"], p["height_m"], p["radius_m"]
        assert y-h/2 == pytest.approx(previous_edge, abs=1e-16)
        previous_edge = y+h/2
        m = wheel.mass_kg*p["mass_fraction"]
        inertia += [m*(3*r*r+h*h)/12 + m*y*y,
                    .5*m*r*r, m*(3*r*r+h*h)/12 + m*y*y]
        mass += m
        first_moment += m*y
        volume += np.pi*r*r*h
    assert previous_edge == pytest.approx(wheel.width_m/2)
    assert volume == pytest.approx(np.pi*wheel.radius_m**2*wheel.width_m)
    assert mass == pytest.approx(wheel.mass_kg)
    assert first_moment == pytest.approx(0., abs=1e-16)
    np.testing.assert_allclose(inertia, wheel.diagonal_inertia_kg_m2, rtol=1e-14)


@pytest.mark.parametrize("changes", [dict(slices=True), dict(slices=9.), dict(slices=2),
    dict(radius_m=0.), dict(width_m=-1), dict(mass_kg=float("nan"))])
def test_invalid_compounds_rejected(changes):
    with pytest.raises(ValueError):
        UniformWheel(**changes)
