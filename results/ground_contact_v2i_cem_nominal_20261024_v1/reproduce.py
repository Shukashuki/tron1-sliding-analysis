"""Read-only nine-case metric/contact replay; no assets, weights or physics."""
import hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/"artifact_manifest.json").read_text())
name="scripts/package_cem_ground_contact_v2i_nominal.py"
assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest["replay_helper_sha256"][name]
from package_cem_ground_contact_v2i_nominal import verify_public
if __name__=="__main__":print(json.dumps(verify_public(HERE),indent=2))
