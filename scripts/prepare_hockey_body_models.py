#!/usr/bin/env python3
"""Extract only required files from user-provided SMPL/SMPL-X archives locally.

No model download, unpickling, upload or redistribution is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive, output, kind):
    records = []
    with zipfile.ZipFile(archive) as z:
        for gender in ("NEUTRAL", "FEMALE", "MALE"):
            if kind == "smpl":
                token = {"NEUTRAL": "neutral", "FEMALE": "f", "MALE": "m"}[gender]
                member = f"SMPL_python_v.1.1.0/smpl/models/basicmodel_{token}_lbs_10_207_0_v1.1.0.pkl"
                name = f"SMPL_{gender}.pkl"
            else:
                name = f"SMPLX_{gender}.npz"
                member = f"models/smplx/{name}"
            destination = output / kind / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Fixed output names; never extract arbitrary ZIP member paths.
            temp = destination.with_suffix(destination.suffix + ".part")
            with z.open(member) as src, temp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if destination.exists():
                if sha(destination) != sha(temp):
                    temp.unlink()
                    raise ValueError(f"Refusing to replace a different model: {destination}")
                temp.unlink()
            else:
                temp.replace(destination)
            records.append({"gender": gender, "archive_member": member,
                            "local_file": str(destination), "bytes": destination.stat().st_size,
                            "sha256": sha(destination)})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smpl", type=Path, required=True)
    p.add_argument("--smplx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    args = p.parse_args()
    records = extract(args.smpl, args.output, "smpl")
    xrecords = extract(args.smplx, args.output, "smplx")
    inventory = {
        "kind": "user_provided_body_model_inventory", "source_archive": str(args.smpl),
        "archive_sha256": sha(args.smpl), "model_type": "SMPL", "SMPLX_present": True,
        "models": records,
        "SMPLX_archive": {"source_archive": str(args.smplx), "archive_sha256": sha(args.smplx),
                          "models": xrecords, "extracted_npz_only": True},
        "weights_not_for_git_or_remote_transfer": True, "unpickling_performed": False,
    }
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(json.dumps(inventory, indent=2) + "\n")


if __name__ == "__main__":
    main()
