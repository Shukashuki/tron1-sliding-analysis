"""Recompute both recorded contact analyses, without actors or new simulation."""
import argparse,json,pathlib,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True)
p.add_argument("--output-dir",type=pathlib.Path,required=True)
a=p.parse_args();root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(a.repository.resolve()/"scripts"))
import analyze_native_contact_dynamics as analysis
from ppo_backend_400hz_contract import read_json,require,sha256
manifest=read_json(root/"manifest.json")
for name,row in manifest["files"].items():
    require(sha256(root/name)==row["sha256"],"Changed published artifact: "+name)
for name,digest in manifest["source_sha256"].items():
    require(sha256(a.repository/name)==digest,"Restore frozen sources in a fresh checkout: "+name)
analysis.BASELINES={tag:root/(tag+"_native_v1/physics_trajectory.npz") for tag in ("k","n")}
actual=analysis.run(argparse.Namespace(collection=root,output_dir=a.output_dir))
expected=read_json(root/"analysis.json")
require(actual["rows"]==expected["rows"],"Contact, command or event metrics do not reproduce")
with np.load(root/"native_contact_channels.npz",allow_pickle=False) as x, \
     np.load(a.output_dir/"native_contact_channels.npz",allow_pickle=False) as y:
    require(x.files==y.files,"Different native signal names")
    for name in x.files:
        require(np.array_equal(x[name],y[name],equal_nan=True),"Native signal differs: "+name)
print(json.dumps({"both_recorded_contact_analyses_exact":True,"all_native_channels_exact":True,
"new_physics_training_or_inference":False,"causality_claimed":False,"goal_complete":False}))
