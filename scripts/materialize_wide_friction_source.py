"""Create an isolated evaluation source with equal friction supported to 2.0.

Copies only scripts/training/config, preserving the supplied source checkout.
The frozen-policy and training-profile checks remain intact; only validation
coefficient bounds are extended. Both engines retain their material readbacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from materialize_friction_evaluator import generate


DRIFT_HASHES = {
    "d5da59c7af58f8a45e3deeb8f69fb0af8d382335e7b0974a5637106fc8c56bb6",
    "5eb40c9b3c14f5bd5c59c9861c1414ae8ac1b219ff0dd70591f2c6fe0bb5344c",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extend_sources(source: Path) -> tuple[str, str]:
    drift = source / "training/tron1_drift.py"
    if sha(drift) not in DRIFT_HASHES:
        raise ValueError("Drift module differs from the pinned public/formal source")
    text = drift.read_text()
    old = '("contact_friction", .01, 1.)'
    if text.count(old) != 1:
        raise ValueError("Unexpected drift friction validation anchor")
    drift_text = text.replace(old, '("contact_friction", .01, 2.)', 1)
    evaluator = generate(source / "scripts/eval_tracking.py")
    for old, new in (
        ('.01 <= args.evaluation_contact_friction <= 1.',
         '.01 <= args.evaluation_contact_friction <= 2.'),
        ('finite and in [0.01, 1.0]', 'finite and in [0.01, 2.0]'),
    ):
        if evaluator.count(old) != 1:
            raise ValueError("Unexpected evaluator bound anchor")
        evaluator = evaluator.replace(old, new, 1)
    return drift_text, evaluator


def materialize(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing source snapshot: {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Evaluation source must be separate from its source checkout")
    drift, evaluator = extend_sources(source)
    source_hashes = {str(p.relative_to(source)): sha(p)
                     for directory in ("scripts", "training", "config")
                     if (source / directory).is_dir()
                     for p in sorted((source / directory).rglob("*"))
                     if p.is_file() and p.suffix in (".py", ".json")}
    output.mkdir(parents=True)
    for directory in ("scripts", "training", "config"):
        if (source / directory).is_dir():
            shutil.copytree(source / directory, output / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt", "*.npz"))
    (output / "training/tron1_drift.py").write_text(drift)
    (output / "scripts/eval_tracking_friction_isaac.py").write_text(evaluator)
    changed = ["training/tron1_drift.py", "scripts/eval_tracking_friction_isaac.py"]
    audit = {"schema_version": 1, "purpose": "isolated_fixed_friction_validation",
             "evaluation_mu_bounds": [0.01, 2.0], "training_unchanged": True,
             "source": str(source), "source_hashes": source_hashes,
             "changed_output_sha256": {name: sha(output / name) for name in changed},
             "materializer_sha256": sha(Path(__file__))}
    (output / "wide_friction_source.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source_root, args.output_dir)
    print(json.dumps({"status": "materialized", "output": str(args.output_dir),
                      "evaluation_mu_bounds": [0.01, 2.0]}))


if __name__ == "__main__":
    main()
