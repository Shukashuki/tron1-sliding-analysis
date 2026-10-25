"""Read-only four-case O PPO native metric replay; no physical assets or actors."""
import argparse,copy,json,pathlib,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True)
a=p.parse_args();root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(a.repository.resolve()/"scripts"))
import analyze_ppo_ground_contact_v2o as audit
from ppo_backend_400hz_contract import read_json,require,sha256
manifest=read_json(root/"manifest.json")
for name,row in manifest["files"].items():
    require(sha256(root/name)==row["sha256"],"Changed published artifact: "+name)
for name,digest in manifest["source_sha256"].items():
    require(sha256(a.repository/name)==digest,"Restore frozen source in a fresh checkout: "+name)
with np.load(root/"native_states.npz",allow_pickle=False) as archive:
    actual=audit.recompute_compact({name:archive[name].copy() for name in archive.files})
summary=read_json(root/"analysis.json");protocol=read_json(root/"protocol.json");screens={}
for engine,row in zip(audit.BACKENDS,summary["rows"],strict=True):
    for variant,field in (("new","new_V2"),("baseline","original_grounded_baseline")):
        expected=copy.deepcopy(row[field]);expected.pop("support_force_source",None)
        require(actual[variant+"_"+engine]==expected,"Four-case native metrics differ: "+variant+engine)
    screen=audit.nominal_screen(row["new_V2"],row["original_grounded_baseline"],protocol,engine)
    require(screen==row["nominal_screen"],"Frozen nominal screen differs: "+engine)
    screens[engine]={"pass":screen["numeric_nominal_episode_pass"],"failed_checks":screen["failed_checks"]}
print(json.dumps({"four_case_native_metrics_exact":True,"screens":screens,
"physics_policy_inference_or_training_run":False,"overall_task_complete_claimed":False}))
