"""Replace only the verified waiting dispatcher, then start on GPU 1."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

root=Path('/data/zengxiong/workspaces/tron1-hockey-bilateral-20261028-v1')
bundle=root/'source_snapshot_v1'
script=root/'execution_snapshot_v2/dispatch_hockey_bilateral_capacity.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='a524e74af9a028edd027b8825993b799f9f33c36cb22c628e0aa5124e354fa32'
spec=importlib.util.spec_from_file_location('capacity',script)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prior=Path('/data/zengxiong/workspaces/tron1-physx-reference-generation-20261025-v1/validation_v1.launch.json')
helper_path=Path('/data/zengxiong/workspaces/tron1-ground-contact-v2o-20261025/continuation_snapshot_v1/scripts/continue_ground_contact_v2o_validation.py')
plan,cfg,paths,python=m.inspect(bundle,prior,helper_path)
spec=importlib.util.spec_from_file_location('gpu',helper_path)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.GPU_ORDER=(1,)
gpu=m.probe_available_device(h,1,plan['dependencies'])
assert gpu is not None,'GPU 1 no longer qualifies'
old=json.loads((root/'launch.json').read_text())
state=json.loads((root/'run_v1/status.json').read_text())
assert state['status']=='waiting_for_existing_studies' and state['gpu_jobs_started']==0 and state['current'] is None
pid=old['pid'];proc=Path('/proc')/str(pid)
assert proc.stat().st_uid==os.getuid()
argv0=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
already_exited=h.process_snapshot(pid)
zombie=(already_exited['state']=='Z' and pid==4032578 and already_exited['start_ticks']==2063181838)
assert zombie or argv0==old['argv']
assert not (proc/'task'/str(pid)/'children').read_text().strip()
assert not (root/'run_v2').exists() and not (root/'immediate_launch.json').exists()
session='hockey_bilateral_20261028_v2'
assert subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
identity=h.process_snapshot(pid)
current=h.process_snapshot(pid)
assert all(current[k]==identity[k] for k in ('uid','argv','start_ticks'))
if not zombie:os.kill(pid,signal.SIGTERM)
for _ in range(25):
    observed=h.process_snapshot(pid)
    if observed is None or observed['state']=='Z':break
    time.sleep(.2)
observed=h.process_snapshot(pid)
assert observed is None or observed['state']=='Z','Waiting dispatcher has not exited; no new worker launched'
receipt=dict(action='replace_own_waiting_dispatcher_only',old_pid=pid,old_identity=identity,
    old_status=state['status'],old_gpu_jobs_started=0,terminated_active_training=False,
    device_probe=gpu,time_unix_s=time.time(),terminal_process_state=observed,
    original_launch_argv=old['argv'],already_signalled_on_previous_attempt=zombie)
(root/'wait_replacement_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
argv=[python,'-B','-u',str(script),'--bundle',str(bundle),'--prior-launch',str(prior),
      '--audit-helper',str(helper_path),'--output',str(root/'run_v2'),'--physical-gpu','1']
command='exec '+shlex.join(argv)+' > '+shlex.quote(str(root/'immediate_dispatcher.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(root),command],check=True)
time.sleep(4)
newpid=int(subprocess.check_output(['tmux','list-panes','-t',session,'-F','#{pane_pid}'],text=True).strip())
launch=dict(session=session,pid=newpid,identity=h.process_snapshot(newpid),argv=argv,physical_gpu=1,
    training_manifest_sha256=hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),
    dispatcher_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),created_unix_s=time.time())
(root/'immediate_launch.json').write_text(json.dumps(launch,indent=2)+'\n')
status=json.loads((root/'run_v2/status.json').read_text())
print(json.dumps(dict(replacement=receipt,launch=launch,status=status)))
