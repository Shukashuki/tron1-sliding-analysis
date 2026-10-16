from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from terminal_observation import merge_bootstrap_observations


def test_bootstrap_replaces_only_done_rows_without_mutating_reset_observations():
    after_reset = {"policy": np.array([[10., 11.], [20., 21.], [30., 31.]]),
                   "critic": np.array([[1.], [2.], [3.]])}
    final = {"policy": np.array([[90., 91.]]), "critic": np.array([[9.]])}
    merged = merge_bootstrap_observations(after_reset, np.array([1]), final)
    np.testing.assert_array_equal(merged["policy"], [[10., 11.], [90., 91.], [30., 31.]])
    np.testing.assert_array_equal(merged["critic"], [[1.], [9.], [3.]])
    np.testing.assert_array_equal(after_reset["policy"][1], [20., 21.])


def test_bootstrap_rejects_missing_privileged_observation_or_wrong_batch():
    after_reset = {"policy": np.zeros((3, 2)), "critic": np.zeros((3, 4))}
    with pytest.raises(ValueError, match="groups"):
        merge_bootstrap_observations(after_reset, [1], {"policy": np.zeros((1, 2))})
    with pytest.raises(ValueError, match="shape"):
        merge_bootstrap_observations(after_reset, [1], {"policy": np.zeros((3, 2)), "critic": np.zeros((1, 4))})
