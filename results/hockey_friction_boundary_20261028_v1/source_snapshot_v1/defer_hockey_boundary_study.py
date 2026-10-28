"""Wait for the approved 18-cell study, then run one serial boundary ablation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from dispatch_hockey_speed_study import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-status','bundle','prior-launch','audit-helper','output','status'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    if a.status.exists() or a.output.exists():
        raise FileExistsError('Never duplicate a deferred experiment')
    a.status.parent.mkdir(parents=True,exist_ok=True)
    state = dict(status='waiting_for_parent', started_unix_s=time.time(),
                 parent_status=str(a.parent_status), bundle=str(a.bundle))
    def save():
        state['updated_unix_s']=time.time();write(a.status,state)
    try:
        manifest = json.loads((a.bundle/'manifest.json').read_text())
        for f in manifest['files']:
            path=(a.bundle/f['path']).resolve()
            if not path.is_relative_to(a.bundle.resolve()) or sha(path)!=f['sha256']:
                raise ValueError('Payload changed')
        while True:
            parent=json.loads(a.parent_status.read_text())
            if parent['manifest_sha256']!='a61d76d6a7003d874be48cb3304cb551a0e05c66afbcd51f1eed2b2d4e548197':
                raise ValueError('Unexpected parent study')
            state['parent_stage']=parent.get('current');save()
            if parent['status']=='completed_all_18_cells':
                if len(parent['completed'])!=72 or parent.get('current') is not None:
                    raise ValueError('Parent completion is incomplete')
                break
            if parent['status']=='stopped' or time.time()-state['started_unix_s']>86400:
                raise RuntimeError('Parent stopped or 24-hour dependency wait expired; no child jobs launched')
            time.sleep(30)
        argv=[sys.executable,'-B','-u',str(a.bundle/'dispatch_hockey_speed_study.py')]
        for name in ('bundle','prior_launch','audit_helper','output'):
            argv.extend(['--'+name.replace('_','-'),str(getattr(a,name))])
        state.update(status='dispatching_serial_ablation',argv=argv);save()
        result=subprocess.run(argv,check=False)
        if result.returncode:
            raise RuntimeError(f'Boundary dispatcher returned {result.returncode}; no retry')
        state['status']='completed';save()
    except BaseException as exc:
        state.update(status='stopped',error=f'{type(exc).__name__}: {exc}');save()
        raise


if __name__=='__main__':
    main()
