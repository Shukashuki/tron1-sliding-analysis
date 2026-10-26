from pathlib import Path
import json,hashlib,importlib.util,subprocess,os,time
r=Path('/data/zengxiong/workspaces/tron1-physx-reference-generation-20261025-v1')
p=Path('/data/zengxiong/workspaces/tron1-ground-contact-v2o-20261025/continuation_snapshot_v1/scripts/continue_ground_contact_v2o_validation.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='6f5285f9e9515d98ea4ae6e99c3bf58a436fd98a3ca266e1f297f4501cce3866'
s=importlib.util.spec_from_file_location('audit',p);h=importlib.util.module_from_spec(s);s.loader.exec_module(h);h.GPU_ORDER=(3,5,4,0,1,2,6,7)
old=json.loads((r/'search_v1.launch.json').read_text());prior=h.process_snapshot(old['pid'])
assert prior is None or prior['state']=='Z' or prior['start_ticks']!=old['identity']['start_ticks']
report=json.loads((r/'search_v1/report.json').read_text())
assert report['steps']==349 and report['iterations']==4
m=json.loads((r/'source_snapshot_v2/source_manifest.json').read_text())
assert all(hashlib.sha256((r/'source_snapshot_v2'/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in m['files'])
subprocess.run(['/data/zengxiong/.conda/envs/regrind/bin/python','-c',"import numpy as np;np.savez_compressed("+repr(str(r/'zero_mean.npz'))+",latent_mean=np.zeros((12,8)))"],check=True)
for name,mean in [('validation_v1',r/'search_v1/mean.npz'),('baseline_validation_v1',r/'zero_mean.npz')]:
 idle=next(x for x in h.query_gpus() if x['idle'])
 argv=[x.replace('/search_v1', '/'+name) for x in old['argv']]
 argv[argv.index('--device')+1]='cuda:'+str(idle['physical_index'])
 argv+=['--validate-mean',str(mean)]
 with (r/(name+'.log')).open('xb') as log:
  child=subprocess.Popen(argv,cwd=r,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID'),shell=False)
 i=h.process_snapshot(child.pid);assert i and i['uid']==os.getuid() and i['argv']==argv
 (r/(name+'.launch.json')).write_text(json.dumps(dict(pid=child.pid,argv=argv,identity=i,gpu_before_launch=idle),indent=2)+'\n')
 code=child.wait()
 assert code==0 and not (r/name/'failure.json').exists(), name+' failed; preserve evidence'
 result=json.loads((r/name/'report.json').read_text());assert result['single_world_validation'] and result['steps']==349
 print(name,'completed',result['metrics'],flush=True)
(r/'validation_chain_completed.json').write_text(json.dumps(dict(status='completed_two_serial_single_world_validations',process_signals_sent=False,automatic_retry=False),indent=2)+'\n')
