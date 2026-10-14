"""Verify a frozen-policy friction grid and render only top-down XY paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_saved_xy as xy


def build(manifest_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("policy_training_friction_randomization") is not True:
        raise ValueError("Grid manifest must declare a newly friction-randomized training policy")
    training_manifest = Path(manifest["training_manifest"])
    trained = json.loads(training_manifest.read_text())
    recipe = trained.get("recipe", {})
    if (recipe.get("friction_static") != [0.3, 1.6]
            or recipe.get("friction_dynamic") != [0.3, 1.2]
            or recipe.get("material_buckets") != 64):
        raise ValueError("Training manifest lacks the pinned BeyondMimic material recipe")
    ref_file = Path(manifest["reference_file"])
    ref = xy.arrays(ref_file)
    rt = np.asarray(ref["time_s"], dtype=float)
    rp = np.asarray(ref["root_target_position_m"][:, :2], dtype=float)
    xy.validate_path(rt, rp, "reference")
    trace = {"reference_t": rt, "reference_xy": rp}
    provenance = {"training_manifest_sha256": xy.sha(training_manifest),
                  "reference_sha256": xy.sha(ref_file), "cases": {}}
    cases = {}
    checkpoint_hashes = set()
    motion_hashes = set()
    for item in manifest["cases"]:
        name = item["case"]
        mu = float(item["mu"])
        if name in cases or not np.isfinite(mu) or mu <= 0:
            raise ValueError("Duplicate/invalid grid case")
        cases[name] = (mu, "")
        ia_file, mj_file = Path(item["isaac_physics"]), Path(item["mujoco_rollout"])
        ia_report_file, mj_report_file = Path(item["isaac_report"]), Path(item["mujoco_report"])
        ia_report = json.loads(ia_report_file.read_text())
        mj_report = json.loads(mj_report_file.read_text())
        override = ia_report["evaluation_contact_friction_override"]
        if (not np.isclose(override["evaluation_contact_friction"], mu)
                or not np.isclose(ia_report["contact_material_audit"]["expected_friction"], mu)
                or not np.allclose(mj_report["contact_material_audit"]["compiled_friction_range"], [mu, mu])
                or ia_report["physics_trajectory_sha256"] != xy.sha(ia_file)
                or ia_report["motion_file_sha256"] != mj_report["motion_sha256"]):
            raise ValueError(f"{name}: friction/material/trace/reference audit failed")
        checkpoint_hashes.add(ia_report["checkpoint_sha256"])
        motion_hashes.add(ia_report["motion_file_sha256"])
        it, ip = xy.observed_episode(xy.arrays(ia_file), "isaac")
        mt, mp = xy.observed_episode(xy.arrays(mj_file), "mujoco")
        trace.update({f"{name}_isaac_t": it, f"{name}_isaac_xy": ip,
                      f"{name}_mujoco_t": mt, f"{name}_mujoco_xy": mp})
        provenance["cases"][name] = {
            "mu": mu, "isaac_trace_sha256": xy.sha(ia_file), "mujoco_trace_sha256": xy.sha(mj_file),
            "isaac_report_sha256": xy.sha(ia_report_file), "mujoco_report_sha256": xy.sha(mj_report_file),
            "isaac_material_readback": ia_report["contact_material_audit"],
            "mujoco_compiled_friction_range": mj_report["contact_material_audit"]["compiled_friction_range"],
        }
    if not cases or len(checkpoint_hashes) != 1 or len(motion_hashes) != 1:
        raise ValueError("Grid must reuse one checkpoint and one motion")
    if next(iter(checkpoint_hashes)) != xy.sha(Path(manifest["checkpoint"])):
        raise ValueError("Grid checkpoint hash differs from actual frozen checkpoint")
    if next(iter(motion_hashes)) != trained["motion_sha256"]:
        raise ValueError("Grid motion differs from policy training")
    provenance["checkpoint_sha256"] = checkpoint_hashes.pop()
    provenance["motion_sha256"] = motion_hashes.pop()
    config = json.loads((Path(__file__).resolve().parents[1] / "config/evaluation_friction.json").read_text())
    result = xy.summarize(trace, provenance, config, cases=cases)
    result["kind"] = "new_beyondmimic_dr_policy_fixed_friction_xy_grid"
    result["policy_training_friction_randomization"] = True
    result["limitations"] = [x for x in result["limitations"] if "archived" not in x and "mu=0.15" not in x]
    result["limitations"].append("One deterministic episode per coefficient; the friction failure boundary needs more seeds or a finer grid.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserving nonempty analysis output")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "xy_trace.npz", **trace)
    (output / "xy_trace.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    xy.render(trace, output / "topdown.png", cases=cases)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.manifest, args.output_dir)
    print(json.dumps({"status": "analyzed", "cases": len(result["rows"]),
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
