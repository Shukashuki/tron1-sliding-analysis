from pathlib import Path
import sys,json,hashlib,numpy as np
r=Path(__file__).resolve().parents[2];sys.dont_write_bytecode=True;sys.path.insert(0,str(r/'scripts'))
import ground_contact_reward_v2h as h
p=r/'results/ground_contact_v2g_review_20261023_v1/phase_inputs.npz'
with np.load(p,allow_pickle=False) as z:data={name:z[name].copy() for name in z.files}
inputs={}
for label in ('A','F','G'):
 for engine in ('physx','mujoco'):
  case=label+'__new_'+engine+'__'
  assert np.allclose(data[case+'time_s'],np.arange(2793)*.0025,atol=1e-9,rtol=0)
  inputs[label+'_PPO_'+engine]=(data[case+'floor_normal_force'][...,2],data[case+'com_velocity'][:,2])
cem=r/'results/ground_contact_v2g_cem_nominal_20261023_v1/native_comparison_signals.npz'
with np.load(cem,allow_pickle=False) as z:
 inputs['G_CEM_mujoco']=(z['actual_V2g__ground_normal_Fz_n'].copy(),z['actual_V2g__com_velocity'][:,2].copy())
rows=[];windows=[('core',1.44,4.),('post_turn',4.,5.86),('stop',5.86,6.96)]
for case,(force,velocity) in inputs.items():
 assert force.shape==(2793,2) and velocity.shape==(2793,)
 comp=h.normal_motion_penalty_numpy(force[1:].reshape(349,8,2),velocity[1:].reshape(349,8))
 start=np.arange(349)*.02;values=[]
 for phase,begin,end in windows:
  selected=(start>=begin-1e-9)&(start<end-1e-9)
  values.append({'phase':phase,'control_windows':int(selected.sum()),
   'mean_added_raw_penalty':float(comp['loaded_com_vz_penalty'][selected].mean()),
   'affected_window_fraction':float(np.mean(comp['loaded_com_vz_penalty'][selected]>0.))})
 rows.append({'case':case,'phases':values})
HERE=Path(__file__).resolve().parent
value=json.loads((HERE/'added_cost_proxy.json').read_text())
assert hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()==value['definition_sha256']
for name,digest in value['public_input_sha256'].items():
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest
assert rows==value['rows']
print(json.dumps({'status':'H_ADDED_COST_ONLY_SEVEN_SAVED_CASES_EXACT','cases':len(rows),
 'H_runtime_or_policy_trained':False,'models_weights_simulation_or_original_outputs_required':False,
 'H_task_success_claimed':False},indent=2))

