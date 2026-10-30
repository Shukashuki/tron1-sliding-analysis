"""Prevent time warping or showing future source poses in a comparison."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_hockey_human_to_policy import timeline


def test_source_and_robot_share_elapsed_seconds_not_normalized_duration():
    source=np.arange(65)/30
    t,k,held=timeline(source,.0025,2.76)
    assert len(t)==1105 and len(t[::8])==139
    assert k[400]==30  # At one second, both show one second.
    assert k[848]==63  # 2.120 s precedes source sample 64 at 2.133 s.
    assert k[856]==64 and held[856]  # 2.140 s: explicit end hold.
    assert k[-1]==64 and held[-1]
    assert np.all(source[k]<=t+1e-10)
    assert np.max((t-source[k])[~held])<1/30+1e-10


def test_common_sample_is_not_rounded_back_or_marked_held():
    t,k,held=timeline(np.arange(7)/30,.0025,.25)
    assert k[40]==3 and not held[40]
    assert k[80]==6 and not held[80]
    assert k[81]==6 and held[81]


def test_reject_ambiguous_clocks():
    for source,dt,horizon in [([.1,.2],.0025,.3),([0,.1,.1],.0025,.3),
                              ([0,.1],0,.3),([0,.1],.0025,.301)]:
        with pytest.raises(ValueError):
            timeline(source,dt,horizon)
