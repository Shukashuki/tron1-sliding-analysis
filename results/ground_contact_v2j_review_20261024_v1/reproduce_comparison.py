"""Reproduce six rows from SHA-pinned public analyses; no assets or simulation.

This verifies the comparison table, not a new native-metric calculation.
Each underlying bundle provides its own recorded-state numerical replay.
"""
from pathlib import Path
import csv
import hashlib
import io
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = [
    {
        "candidate": "I",
        "controller": "PPO",
        "path": "results/ground_contact_v2i_ppo_nominal_20261024_v1/analysis.json",
        "sha256": "c72d227b9a4e12d5faebba079b684c96ff201a29c4cde0573ace515f46cfce75"
    },
    {
        "candidate": "I",
        "controller": "CEM",
        "path": "results/ground_contact_v2i_cem_nominal_20261024_v1/cem_analysis.json",
        "sha256": "9730d7f3a29d1c863a8fea9271318ec3f54d0ff07410954a9d3da54c6da65c79"
    },
    {
        "candidate": "J",
        "controller": "PPO",
        "path": "results/ground_contact_v2j_ppo_nominal_20261024_v1/analysis.json",
        "sha256": "45fe45720d1ff112485a701175f25c6f6b5ad877fd796d21f99ec478d381cbc5"
    },
    {
        "candidate": "J",
        "controller": "CEM",
        "path": "results/ground_contact_v2j_cem_nominal_20261024_v2/cem_analysis.json",
        "sha256": "30414df33c9c44a557dc82db3d5cdabcf6b960f4858f35c02362e05ba62b23d9"
    }
]
FIELDS = ("candidate", "controller", "validation_backend", "core_xy_rmse_m",
    "whole_xy_rmse_m", "whole_xy_peak_m", "core_beta_rmse_deg", "core_speed_retention",
    "whole_anywheel_support_fraction", "whole_longest_no_support_s",
    "core_HF_ratio_to_original_same_controller_engine", "numeric_nominal_episode_pass",
    "failed_checks")

def comparison_rows():
    rows = []
    for source in INPUTS:
        path = ROOT / source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"], path
        data = json.loads(path.read_text())
        assert data["status"] == "completed_analysis"
        if source["controller"] == "PPO":
            cases = [(r["validation_backend"], r["new_V2"], r["nominal_screen"])
                     for r in data["rows"]]
        else:
            cases = [("native CPU MuJoCo", data["actual_V2"+source["candidate"].lower()],
                      data["nominal_screen"])]
        for engine, metrics, screen in cases:
            core = metrics["windows"]["core"]["tracking"]
            whole = metrics["windows"]["whole_reference_overlap"]["tracking"]
            contact = metrics["any_wheel_ground_support"]["full_contact"]
            ratio = screen["core_HF_ratio_to_original_same_engine_PPO"
                           if source["controller"] == "PPO" else "core_HF_ratio_to_old_CEM"]
            rows.append(dict(zip(FIELDS, (source["candidate"], source["controller"], engine,
                core["xy_rmse_m"], whole["xy_rmse_m"], whole["xy_peak_m"],
                core["beta_rmse_deg"], core["speed_retention"],
                contact["any_wheel_support_sample_fraction"],
                contact["longest_no_support_observed_s"], ratio,
                screen["numeric_nominal_episode_pass"], "|".join(screen["failed_checks"])),
                strict=True)))
    assert len(rows) == 6
    return rows

def comparison_csv(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()

if __name__ == "__main__":
    assert comparison_csv(comparison_rows()) == (HERE/"comparison.csv").read_bytes()
    print(json.dumps({"status": "SIX_ACTUAL_PUBLIC_ANALYSIS_ROWS_REPRODUCED_EXACT",
        "native_metrics_recomputed_here": False, "assets_or_weights_required": False,
        "simulation_inference_training_or_file_writes": False, "goal_complete": False}))
