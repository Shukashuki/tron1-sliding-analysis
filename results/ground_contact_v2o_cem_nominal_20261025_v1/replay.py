"""Read-only replay of O metrics and all349 recorded stage rewards."""
import argparse,json,pathlib,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True)
a=p.parse_args();root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(a.repository.resolve()/"scripts"))
import analyze_cem_ground_contact_v2o as audit
from ppo_backend_400hz_contract import read_json,require,sha256
manifest=read_json(root/"manifest.json")
for name,row in manifest["files"].items():
    require(sha256(root/name)==row["sha256"],"Changed published artifact: "+name)
for name,digest in manifest["source_sha256"].items():
    require(sha256(a.repository/name)==digest,"Restore frozen source in a fresh checkout: "+name)
with np.load(root/"actual/rollout.npz",allow_pickle=False) as f:
    raw={name:f[name].copy() for name in f.files}
summary=read_json(root/"analysis.json");report=read_json(root/"actual/report.json")
decisions=json.loads((root/"actual/solver_log.json").read_text())
reference=audit.base.metrics.reference_data(root/"reference.npz")
actual=audit.base.recorded_metrics(raw,reference)
require(actual==summary["actual_V2o"],"Native O metrics do not reproduce")
reward=audit.runner.recompute_actual_rewards(raw,report,decisions,reference_file=root/"reference.npz")
require(reward==summary["actual_O_reward_audit"],"Recorded15-term rewards do not reproduce")
screen=audit.base.nominal_screen(actual,summary["original_grounded_baseline"],read_json(root/"protocol.json"))
require(screen==summary["nominal_screen"],"Frozen nominal screen does not reproduce")
print(json.dumps({"current_O_metrics_exact":True,"recorded_stage_rewards_exact":True,"stages":349,
"numeric_nominal_episode_pass":screen["numeric_nominal_episode_pass"],"failed_checks":screen["failed_checks"],
"physics_policy_inference_or_training_run":False,"overall_task_complete_claimed":False}))
