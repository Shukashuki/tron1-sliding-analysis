"""Force-spring units and fixed-profile identities, not simulation success."""
from pathlib import Path
import math
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from physx_normal_compliance import profile


def test_reference_deflection_and_damping_have_force_spring_units():
    mass=11.1365
    p=profile("compliant_1mm_d1",mass)
    assert p["stiffness_n_m"]*.001==pytest.approx(mass*9.81)
    assert p["damping_n_s_m"]/(2*math.sqrt(p["stiffness_n_m"]*mass))==pytest.approx(1.)
    assert p["acceleration_spring"] is False
    q=profile("compliant_3mm_d1",mass)
    assert q["stiffness_n_m"]==pytest.approx(p["stiffness_n_m"]/3)
    assert q["damping_n_s_m"]==pytest.approx(p["damping_n_s_m"]/math.sqrt(3))


def test_zero_damping_control_keeps_exactly_the_same_stiffness():
    p,q=profile("compliant_3mm_d1",11.1365),profile("compliant_3mm_d0",11.1365)
    assert p["stiffness_n_m"]==q["stiffness_n_m"] and q["damping_n_s_m"]==0.
    assert profile("rigid",11.1365)["stiffness_n_m"]==0.


@pytest.mark.parametrize("mass",(0.,-1.,float("nan"),float("inf")))
def test_invalid_supported_mass_rejected(mass):
    with pytest.raises(AssertionError):profile("compliant_1mm_d1",mass)
