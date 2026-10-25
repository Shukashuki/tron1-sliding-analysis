from __future__ import annotations
import hashlib,json,os,shlex,shutil,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path('/home/roy/tron1-sliding-analysis')
LOCAL=ROOT/'outputs/ground_contact_v2q_20261025_v1'
RESULT=ROOT/'results/ground_contact_v2q_ppo_nominal_20261025_v1'
REMOTE='/data/zengxiong/workspaces/tron1-ground-contact-v2q-20261025'
SSH=['ssh','-i','/home/roy/.ssh/id_ed25519','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','zengxiong@14.103.233.29']
SCP=['scp','-i','/home/roy/.ssh/id_ed25519','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=20']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCES=['scripts/'+n for n in ('collect_ppo_ground_contact_v2q.py','analyze_ppo_ground_contact_v2q.py','render_ppo_ground_contact_v2q.py','verify_ppo_ground_contact_v2q_artifacts.py','ground_contact_reward_v2q.py','wheel_ground_geometry_torch.py')]
LEDGER={n:sha(ROOT/n) for n in SOURCES}
STATE=dict(status='waiting_for_existing_exact_remote_jobs',source_sha256=LEDGER,children=[],task_success_claimed=False)
def write():
 tmp=LOCAL/'finish_local_v1_status.tmp';tmp.write_text(json.dumps(STATE,indent=2)+'\n');os.replace(tmp,LOCAL/'finish_local_v1_status.json')
def remote_python(program):
 result=subprocess.run(SSH+[shlex.join(['/usr/bin/python3','-B','-c',program])],check=True,capture_output=True,text=True,timeout=55)
 return json.loads(result.stdout)
def ensure_sources():
 assert all(sha(ROOT/n)==v for n,v in LEDGER.items())
def child(name,argv):
 ensure_sources();STATE['status']='running_'+name;write()
 with (LOCAL/(name+'_local_v1.log')).open('xb') as stream:
  p=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'MUJOCO_GL':'osmesa','PYTHONDONTWRITEBYTECODE':'1'})
  row=dict(name=name,pid=p.pid,argv=argv,status='running');STATE['children'].append(row);write()
  code=p.wait()
 row.update(returncode=code,status='completed' if code==0 else 'failed');write()
 assert code==0,('Local stage failed, no retry',name,code)
def main():
 assert not (LOCAL/'finish_local_v1_status.json').exists()
 write()
 while True:
  program="""import json,time
from pathlib import Path
r=Path('"""+REMOTE+"""');out=dict(time=time.time(),processes={})
for key,name,start in (('driver','driver_launch.json',2031292224),('formal','formal_v1_launch.json',2031294344),('nominal','nominal_driver_launch.json',2031298262)):
 record=json.loads((r/name).read_text());p=Path('/proc')/str(record['pid'])
 try:
  s=(p/'stat').read_text().rsplit(') ',1)[1].split();assert int(s[19])==start
  argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
  assert argv==record['argv'] or s[0]=='Z'
  out['processes'][key]=dict(pid=record['pid'],state=s[0],start_ticks=int(s[19]))
 except FileNotFoundError:out['processes'][key]=None
for key,name in (('training','driver_status.json'),('nominal','nominal_v1/status.json')):
 d=json.loads((r/name).read_text());out[key]=d['status']
print(json.dumps(out))"""
  # Observation failure is not proof that any remote process ended.
  try:observed=remote_python(program)
  except subprocess.TimeoutExpired as error:
   STATE['last_observation_timeout']=str(error);write();time.sleep(30);continue
  STATE['last_remote_observation']=observed;write();print(json.dumps(observed),flush=True)
  assert not observed['training'].startswith('failed') and not observed['nominal'].startswith('failed')
  if observed['nominal']=='both_nominal_engines_completed_require_analysis':
   if all(p is None or p['state']=='Z' for p in observed['processes'].values()):break
  time.sleep(30)
 ensure_sources()
 collector=REMOTE+'/collect_ppo_ground_contact_v2q.py'
 info=remote_python("import hashlib,json,subprocess;from pathlib import Path;p=Path("+repr(collector)+");assert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(LEDGER['scripts/collect_ppo_ground_contact_v2q.py'])+";r=subprocess.run(['/usr/bin/python3','-B',str(p),'--workspace',"+repr(REMOTE)+"],check=True,capture_output=True,text=True);print(r.stdout)")
 STATE.update(status='collecting_completed_native_evidence',remote_collection=info);write()
 archive=LOCAL/'completed_collection_v1.tar.gz'
 subprocess.run(SCP+['zengxiong@14.103.233.29:'+info['archive'],str(archive)],check=True,timeout=180)
 assert sha(archive)==info['sha256']
 assert not RESULT.exists();RESULT.mkdir()
 with tarfile.open(archive) as tar:
  members=tar.getmembers()
  assert len({m.name for m in members})==len(members)
  assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
  manifest=json.load(tar.extractfile('collection_manifest.json'))
  assert manifest['kind']=='completed_Q_dual_engine_native_evidence'
  assert {m.name for m in members}==set(manifest['files'])|{'collection_manifest.json'}
  for name,row in manifest['files'].items():
   content=tar.extractfile(name).read();assert hashlib.sha256(content).hexdigest()==row['sha256'] and len(content)==row['bytes']
  tar.extractall(RESULT/'evidence',filter='data')
 baseline=ROOT/'results/ground_contact_v2p_ppo_nominal_20261025_v1/baseline_K'
 shutil.copytree(baseline,RESULT/'baseline_K')
 common=['--collection',str(RESULT/'evidence'),'--k-raw-root',str(RESULT/'baseline_K')]
 child('analysis',[sys.executable,'-B','scripts/analyze_ppo_ground_contact_v2q.py',*common,'--output-dir',str(RESULT/'analysis')])
 child('render',[sys.executable,'-B','scripts/render_ppo_ground_contact_v2q.py',*common,'--analysis',str(RESULT/'analysis/analysis.json'),'--output-dir',str(RESULT/'render')])
 child('artifact_verification',[sys.executable,'-B','scripts/verify_ppo_ground_contact_v2q_artifacts.py','--result',str(RESULT)])
 STATE.update(status='ready_for_visual_review',result=str(RESULT),task_success_claimed=False);write()
 print(json.dumps(STATE),flush=True)
try:main()
except BaseException as error:
 STATE.update(status='failed_no_automatic_restart',error=repr(error));write();raise
