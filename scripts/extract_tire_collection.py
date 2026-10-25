"""Extract an explicitly SHA-bound numeric diagnostic collection into a fresh directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def run(a):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(a.manifest)==a.manifest_sha256 and sha(a.archive)==a.archive_sha256
    assert not a.output_dir.exists()
    value=json.loads(a.manifest.read_text());entries={x["path"]:x for x in value["files"]}
    assert value["archive_sha256"]==a.archive_sha256 and len(entries)==len(value["files"])
    assert a.archive.stat().st_size==value["archive_bytes"]
    payload={}
    with tarfile.open(a.archive,"r:gz") as tar:
        members=tar.getmembers()
        assert len(members)==len(entries) and {m.name for m in members}==set(entries)
        assert sum(m.size for m in members)<100000000
        for m in members:
            p=Path(m.name)
            assert m.isfile() and not p.is_absolute() and ".." not in p.parts and p.suffix in (".json",".npz",".gz")
            b=tar.extractfile(m).read()
            assert len(b)==m.size==entries[m.name]["bytes"] and hashlib.sha256(b).hexdigest()==entries[m.name]["sha256"]
            payload[m.name]=b
    a.output_dir.mkdir(parents=True)
    for name,b in payload.items():
        p=a.output_dir/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("xb") as f:f.write(b)
    print(json.dumps({"status":"all_members_verified_before_fresh_extraction","files":len(entries),"target":str(a.output_dir)}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("manifest","archive","output-dir"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("manifest-sha256","archive-sha256"):p.add_argument("--"+n,required=True)
    run(p.parse_args())
