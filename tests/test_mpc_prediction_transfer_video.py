"""Visualization contract tests on actual archived physical recordings.

No fabricated physical episodes are written, rendered, or published.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_mpc_prediction_transfer_video as video


class RecordedTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "results/controller_primary_nmpc_residual_h30_b10_nominal_pair_v1/pair_trace.npz"
        with np.load(path, allow_pickle=False) as source:
            cls.recorded = {name: source[name] for name in source.files}

    def test_reference_and_native_measured_clocks_and_headings(self):
        for label in ("reference", "isaac", "mujoco"):
            trace, signals = video.validate_trace(self.recorded, label + "_", measured=label != "reference")
            self.assertEqual(len(trace["time_s"]), 349 if label == "reference" else 1397)
            self.assertEqual(len(signals["speed_m_s"]), len(trace["time_s"]))

    def test_reject_clock_resampling(self):
        altered = copy.deepcopy(self.recorded)
        altered["isaac_time_s"][10] += .0001
        with self.assertRaisesRegex(ValueError, "native clock"):
            video.validate_trace(altered, "isaac_", measured=True)

    def test_reject_heading_tampering(self):
        altered = copy.deepcopy(self.recorded)
        altered["isaac_body_heading_rad"][10] += .0001
        with self.assertRaisesRegex(ValueError, "heading/speed replay"):
            video.validate_trace(altered, "isaac_", measured=True)

    def test_reject_absent_measured_contacts(self):
        altered = dict(self.recorded)
        del altered["mujoco_contact_count"]
        with self.assertRaisesRegex(ValueError, "Incomplete trace"):
            video.validate_trace(altered, "mujoco_", measured=True)

    def test_preserve_existing_output_before_loading_source(self):
        actual_source_file = ROOT / "results/controller_primary_nmpc_residual_h30_b10_nominal_pair_v1/summary.json"
        args = type("Args", (), {"output_file": actual_source_file, "source_dir": Path("missing")})()
        digest = video.sha256(actual_source_file)
        with self.assertRaisesRegex(ValueError, "Preserving existing"):
            video.run(args)
        self.assertEqual(video.sha256(actual_source_file), digest)

    def test_missing_encoder_fails_explicitly(self):
        with patch.object(video.shutil, "which", return_value=None), patch.dict(sys.modules, {"imageio_ffmpeg": None}):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg is required"):
                video.find_encoder()

    def test_reject_wrong_experiment_without_loading_npz(self):
        with tempfile.TemporaryDirectory() as name:
            source = Path(name)
            with patch.object(Path, "read_text", return_value='{"kind":"not_a_real_matrix"}'):
                with self.assertRaisesRegex(ValueError, "Require the measured"):
                    video.load_recordings(source)


if __name__ == "__main__":
    unittest.main()
