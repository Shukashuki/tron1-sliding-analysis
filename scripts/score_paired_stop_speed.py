"""Common stopping/contact gates for both behaviors; no required lateral slide."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_direct_brake_comparison import analyze_case
from evaluate_hockey_physx import sha


def boundary(counts, cases=33, minimum_passes=30):
    if len(counts)!=2 or any(type(n) is not int or not 0<=n<=cases for n in counts):
        raise ValueError('Require both complete policy scores before selecting a boundary')
    return min(counts)<minimum_passes


def score(suite, reference, reference_report, output):
    meta=json.loads(reference_report.read_text())
    source=json.loads((suite/'report.json').read_text())
    if source['status']!='validation_suite_completed_requires_independent_qualification' or len(source['cases'])!=33:
        raise ValueError('Incomplete suite cannot select a speed')
    rows=[]
    for case in source['cases']:
        directory=suite/case['id']
        if sha(directory/'report.json')!=case['report_sha256']:
            raise ValueError('Suite report changed')
        row,_=analyze_case(directory,reference,meta['budget'],brake_start=meta['brake_start_s'],brake_end=meta['brake_end_s'])
        rows.append(row)
    nominal=next(r for r in rows if r['case_id']=='nominal')
    result=dict(kind='paired_stop_speed_score_v1',scale=meta['scale'],seed=nominal['seed'],
        actor_sha256=nominal['actor_sha256'],cases=33,passes=sum(r['strict_direct_task_pass'] for r in rows),
        bilateral_contact_passes=sum(r['gates']['bilateral_all_native_samples'] for r in rows),
        stop_hold_passes=sum(r['gates']['xyz_stop_and_terminal_hold'] for r in rows),
        median_jitter_rad_s=float(np.median([r['angular_xy_10_100hz_rms_rad_s'] for r in rows])),
        results=rows,suite_report_sha256=sha(suite/'report.json'),reference_report_sha256=sha(reference_report),
        result_is_one_training_seed=True,physics_rerun=False)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('suite','reference','reference-report','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    r=score(a.suite,a.reference,a.reference_report,a.output)
    print(json.dumps({k:r[k] for k in ('scale','seed','cases','passes','bilateral_contact_passes','stop_hold_passes')}))
