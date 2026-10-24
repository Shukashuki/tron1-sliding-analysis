"""Read-only A/G/H/I table replay from pinned public analyses.

No native metrics are recomputed here. Parent bundles provide that proof.
"""
from pathlib import Path
import csv
import hashlib
import io
import json

ROOT = Path(__file__).resolve().parents[2]
PINS = {
    "results/ground_contact_v2_ppo_nominal_20261023_v1/analysis.json": "99b1347e3f34cb0ef875c4495ee5ec6bbf09e9f6e46f11ad1784cb5455acece5",
    "results/ground_contact_v2g_ppo_nominal_20261023_v1/analysis.json": "3eb1202db961e0157fb2d418266357587032bf38eacdf4a1d93d70e34dc23155",
    "results/ground_contact_v2h_ppo_nominal_20261024_v1/analysis.json": "ce0d3400891f0124051092ad0aeb5b26dbf948fccfc94a6a1d4bbd89af3c39a6",
    "results/ground_contact_v2i_ppo_nominal_20261024_v1/analysis.json": "c72d227b9a4e12d5faebba079b684c96ff201a29c4cde0573ace515f46cfce75",
    "results/ground_contact_v2h_cem_nominal_20261024_v1/cem_analysis.json": "fe805fa2831da803d408cce01de9e9d7ed0c5804bd2115eb7925970cbed3b3dc",
    "results/ground_contact_v2i_cem_nominal_20261024_v1/cem_analysis.json": "9730d7f3a29d1c863a8fea9271318ec3f54d0ff07410954a9d3da54c6da65c79",
}
FIELDS = ("variant", "controller", "training_backend", "validation_backend", "core_xy_rmse_m",
          "whole_xy_rmse_m", "whole_xy_peak_m", "core_beta_rmse_deg", "core_speed_ratio",
          "full_anywheel_support_fraction", "unsupported_samples", "longest_gap_s",
          "core_HF_ratio_to_original_same_controller_engine")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    for name, digest in PINS.items():
        assert sha(ROOT/name) == digest, "Public analysis changed: "+name
    return [json.loads((ROOT/name).read_text()) for name in PINS]


def row(variant, controller, training, engine, metrics, hf_ratio):
    c = metrics["windows"]["core"]["tracking"]
    w = metrics["windows"]["whole_reference_overlap"]["tracking"]
    s = metrics["any_wheel_ground_support"]["full_contact"]
    return dict(zip(FIELDS, (variant, controller, training, engine, c["xy_rmse_m"], w["xy_rmse_m"],
        w["xy_peak_m"], c["beta_rmse_deg"], c["speed_retention"], s["any_wheel_support_sample_fraction"],
        s["unsupported_physical_samples"], s["longest_no_support_observed_s"], hf_ratio)))


def build_rows():
    a, g, h, i, cem_h, cem_i = inputs()
    assert cem_h["original_grounded_baseline"] == cem_i["original_grounded_baseline"]
    original_hf = cem_i["original_grounded_baseline"]["windows"]["core"]["world_angular_xy_common10_100"]["common_band_rms"]
    rows = []
    for variant, summary, cem, key in (("A", a, cem_i, "optional_prior_actual_V2a"),
            ("G", g, cem_i, "optional_prior_actual_V2g"), ("H", h, cem_h, "actual_V2h"),
            ("I", i, cem_i, "actual_V2i")):
        for item in summary["rows"]:
            rows.append(row(variant, "PPO", "PhysX", item["validation_backend"], item["new_V2"],
                item["nominal_screen"]["core_HF_ratio_to_original_same_engine_PPO"]))
        m = cem[key]["metrics"] if key.startswith("optional_") else cem[key]
        rows.append(row(variant, "CEM (known mu)", "not learned", "native CPU MuJoCo", m,
            m["windows"]["core"]["world_angular_xy_common10_100"]["common_band_rms"]/original_hf))
    assert len(rows) == 12
    return rows


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def verify():
    folder = Path(__file__).resolve().parent
    manifest = json.loads((folder/"artifact_manifest.json").read_text())
    assert {p.name for p in folder.iterdir()} == {x["path"] for x in manifest["files"]}|{"artifact_manifest.json"}
    for item in manifest["files"]:
        p = folder/item["path"]
        assert not p.is_symlink() and p.stat().st_size == item["bytes"] and sha(p) == item["sha256"]
    assert (folder/"controller_comparison.csv").read_text() == csv_text(build_rows())
    return {"status": "TWELVE_PUBLIC_ANALYSIS_ROWS_EXACT", "native_metric_recomputation_claimed": False,
            "use_parent_bundles_for_native_metric_proof": True,
            "new_physics_inference_models_weights_or_writes": False, "task_complete_claimed": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
