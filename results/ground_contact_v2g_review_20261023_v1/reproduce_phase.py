from pathlib import Path
import json,hashlib,sys,numpy as np
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
reproduction=json.loads((HERE/'reproduction.json').read_text())
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
           for name,digest in reproduction['phase_replay_helper_sha256'].items())
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_ppo_ground_contact_v2g as a
from analyze_cem_ground_contact_v2 import support_summary
r=ROOT
stored=json.loads((HERE/'ppo_phase_diagnostic.json').read_text())
with np.load(HERE/'phase_inputs.npz',allow_pickle=False) as archive:
 common={name:archive[name].copy() for name in archive.files}
rows=[];sources=stored['input_source_hashes']
phases=[('pre_turn',(.02,1.44)),('early_turn',(1.44,2.5)),('late_turn',(2.5,4.)),('post_turn',(4.,5.86)),('stop',(5.86,6.96))]
for label,suffix in [('A',''),('F','f'),('G','g')]:
 data={k.removeprefix(label+'__'):v for k,v in common.items() if k.startswith(label+'__')}
 ref={k.removeprefix('reference_'):v for k,v in common.items() if k.startswith('reference_')}
 for engine in ('physx','mujoco'):
  case='new_'+engine;trace={k.removeprefix(case+'__'):v for k,v in data.items() if k.startswith(case+'__')}
  trace['physics_dt_s']=.0025;a._clock(trace);signals=a._root_signals(trace);time=trace['time_s']
  normal=np.maximum(trace['floor_normal_force'][...,2],0.).sum(-1);load=normal/(22.272998809814453*9.81)
  entry={'variant':label,'engine':engine,'normal_load_p50_p95_p99_max_mg':np.quantile(load,[.5,.95,.99,1.]).tolist(),
   'fraction_normal_load_gt3mg':float(np.mean(load>3.)),
   'windows':[]}
  for name,interval in phases:
   result=a.metrics.window_metrics(trace,ref,signals,interval)
   support=support_summary(time,trace['floor_normal_force'][...,2],interval)
   select=(time>=interval[0]-1e-9)&(time<=interval[1]+1e-9)
   entry['windows'].append({'phase':name,'interval_s':list(interval),'tracking':result['tracking'],
    'support_fraction':support['any_wheel_support_sample_fraction'],'max_gap_s':support['longest_no_support_observed_s'],
    'max_tilt_deg':result['maximum_tilt_deg'],'height_range_m':np.quantile(trace['position'][select,2],[0,1]).tolist(),
    'vertical_velocity_rms_m_s':float(np.sqrt(np.mean(trace['com_velocity'][select,2]**2)))})
  rows.append(entry)
value={'kind':'actual_A_F_G_PPO_phase_diagnostic','input_source_hashes':sources,'rows':rows,'episode_reward_or_causal_physics_claimed':False,
'height_or_tilt_used_as_failure_gate':False,'normal_load_normalization':'descriptive shared PhysX live mass22.272998809814453kg x9.81; not exact reconstructed stage reward',
'no_new_physics_FK_policy_inference_training_or_tuning':True}
assert value==stored,"One or more six-case phase metrics differ"
print(json.dumps({'status':'ALL_SIX_CASE_PHASE_METRICS_EXACT','cases':len(rows),'phases_per_case':len(phases),
 'original_weight_or_model_bytes_reverified':False,'saved_source_fingerprints_preserved':True,
 'new_physics_FK_policy_training_or_file_writes':False,'goal_complete':False},indent=2))

