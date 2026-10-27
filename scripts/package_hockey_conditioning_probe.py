#!/usr/bin/env python3
"""Prepare a reviewable two-case PhysX conditioning diagnostic; never upload/run."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    if a.output.exists():raise FileExistsError(a.output)
    old=root/'results/hockey_stop_physx_probe_20261026_v2/evidence/source_snapshot_v2'
    audit=root/'results/hockey_stop_conditioning_audit_20261027_v1'
    evidence=json.loads((audit/'report.json').read_text())
    preflight=json.loads((audit/'adapter_preflight.json').read_text())
    if preflight['adapter_sha256']!=sha(root/'scripts/hockey_conditioning_adapter.py'):raise ValueError('Adapter changed after preflight')
    if evidence['artifacts']['nominal_conditioning.npz']!=sha(audit/'nominal_conditioning.npz'):raise ValueError('Conditioning table changed')
    config=json.loads((old/'config.json').read_text())
    if config['motion_sha256']!='e23fbd82639d243f0c485cf391b1eff0e275d31f6bb6ba940647874a0ac83050':raise ValueError('Hockey target changed')
    a.output.mkdir(parents=True)
    files=[root/'scripts'/n for n in ('hockey_conditioning_adapter.py','hockey_physx_conditioning_probe.py','dispatch_hockey_conditioning_probe.py')]
    files += [old/n for n in ('motion.npz','command_report.json','config.json','zero_mean.npz')]
    files += [audit/'nominal_conditioning.npz']
    rows=[]
    for source in files:
        dest=a.output/source.name;shutil.copyfile(source,dest)
        rows.append(dict(path=dest.name,source=str(source.relative_to(root)),bytes=dest.stat().st_size,sha256=sha(dest)))
    manifest=dict(kind='hockey_stop_reference_conditioning_probe_v1',files=rows,total_bytes=sum(r['bytes'] for r in rows),
        contains_models_weights_credentials=False,contains_original_RGB=False,
        destination='zengxiong@14.103.233.29:/data/zengxiong/workspaces/tron1-hockey-stop-20261027-v1/source_snapshot_v1/',
        scope='Two serial 139-step native PhysX probes (reference_dq, nominal_command); each one environment on a freshly completely idle RTX; no training/search or changes to other processes',
        physical_target_unchanged_from_approved_v2=True,mu=.7,physics_dt_s=.0025,policy_dt_s=.02,
        approval_status='prepared_not_uploaded_or_run',cpu_audit_report_sha256=sha(audit/'report.json'),cpu_adapter_preflight_sha256=sha(audit/'adapter_preflight.json'))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=a.output.with_suffix('.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for path in sorted(a.output.iterdir()):
            info=tar.gettarinfo(str(path),arcname=path.name)
            info.uid=info.gid=info.mtime=0;info.uname=info.gname=''
            with path.open('rb') as f:tar.addfile(info,f)
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),compressed_bytes=archive.stat().st_size,
        payload_files=len(rows),payload_bytes=manifest['total_bytes'],manifest=str(a.output/'manifest.json')),indent=2))


if __name__=='__main__':main()
