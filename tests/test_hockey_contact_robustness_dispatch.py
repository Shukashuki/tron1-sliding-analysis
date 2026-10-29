"""A plausible success label cannot launch an unqualified/changed policy."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dispatch_hockey_contact_robustness import ready_candidate
from evaluate_hockey_physx import sha


class RobustnessDispatchTests(unittest.TestCase):
    def fixture(self,root):
        bundle=root/'bundle';runs=root/'runs';d=runs/'candidate'
        for p in (bundle,d/'formal',d/'nominal'):p.mkdir(parents=True,exist_ok=True)
        def write(p,value):p.write_text(json.dumps(value))
        cfg=dict(formal=dict(num_envs=512,iterations=250),ppo=dict(steps_per_env=24),
                 warm_start=dict(checkpoint_sha256='parent'))
        write(bundle/'protocol.json',cfg);write(bundle/'manifest.json',{'fixture_only':True})
        actor=d/'formal/actor_normalized.pt';actor.write_bytes(b'identity fixture, not a policy')
        record=d/'nominal/final_rollout.npz';record.write_bytes(b'record identity fixture')
        write(d/'formal/report.json',dict(status='training_completed_not_task_success',transitions=3072000,
            warm_start=dict(checkpoint_sha256='parent'),actor_sha256=sha(actor)))
        write(d/'nominal/report.json',dict(actor_sha256=sha(actor),artifacts={'final_rollout.npz':sha(record)}))
        write(d/'nominal_qualification.json',dict(qualification=dict(task_success_claimed=True,
            gates=dict(bilateral_continuous_contact=True,terminal_stop_hold=True)),actor_sha256=sha(actor),
            source_manifest_sha256=sha(bundle/'manifest.json'),protocol_sha256=sha(bundle/'protocol.json'),
            run_report_sha256=sha(d/'nominal/report.json')))
        write(runs/'status.json',dict(status='nominal_pass_requires_robustness_suite',current=None,
            manifest_sha256=sha(bundle/'manifest.json'),candidate='candidate'))
        return bundle,runs,dict(cells=[dict(id='candidate',protocol='protocol.json')])

    def test_running_and_failed_screens_are_rejected_before_suite_creation(self):
        for mutation in ('running','active_process','failed_gate','false_claim'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                b,r,p=self.fixture(Path(temporary));ready_candidate(b,r,p)
                file=r/'status.json' if mutation in ('running','active_process') else r/'candidate/nominal_qualification.json'
                data=json.loads(file.read_text())
                if mutation=='running':data['status']='running'
                elif mutation=='active_process':data['current']={'pid':123}
                elif mutation=='failed_gate':data['qualification']['gates']['bilateral_continuous_contact']=False
                else:data['qualification']['task_success_claimed']=False
                file.write_text(json.dumps(data))
                with self.assertRaises(ValueError):ready_candidate(b,r,p)

    def test_budget_parent_actor_and_recording_cannot_change_after_nominal_pass(self):
        for mutation in ('budget','parent','actor','record','duplicate_candidate'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                b,r,p=self.fixture(Path(temporary));d=r/'candidate'
                if mutation in ('budget','parent'):
                    file=d/'formal/report.json';data=json.loads(file.read_text())
                    if mutation=='budget':data['transitions']-=24
                    else:data['warm_start']['checkpoint_sha256']='other-parent'
                    file.write_text(json.dumps(data))
                elif mutation=='actor':(d/'formal/actor_normalized.pt').write_bytes(b'changed actor')
                elif mutation=='record':(d/'nominal/final_rollout.npz').write_bytes(b'changed recording')
                else:p['cells'].append(copy.deepcopy(p['cells'][0]))
                with self.assertRaises(ValueError):ready_candidate(b,r,p)


if __name__=='__main__':unittest.main()
