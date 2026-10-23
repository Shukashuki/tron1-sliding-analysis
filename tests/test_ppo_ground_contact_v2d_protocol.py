"""D isolates positive support aggregation; no task/acceptance changes."""
import ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import ground_contact_reward_v2d as math


def test_only_identity_and_positive_aggregation_change_from_C():
    c=json.loads((ROOT/'config/ppo_ground_contact_v2c_400hz_v1.json').read_text());d=json.loads((ROOT/'config/ppo_ground_contact_v2d_400hz_v1.json').read_text())
    assert d['cohort_id']=='tron1_ground_contact_v2d_any_wheel_400hz_v1'
    assert d['reward_definition_version']=='ground_contact_v2d_any_wheel'
    assert d['ground_contact_reward_v2']==math.validate_settings()
    d['cohort_id'],d['reward_definition_version']=c['cohort_id'],c['reward_definition_version']
    d['ground_contact_reward_v2'].update(candidate='ground_contact_v2c',mean_weight=0,minimum_weight=1)
    assert d==c


def test_exact_C_native_force_capture_reward_clock_and_reset_lifecycle():
    trees=[ast.parse((ROOT/'scripts'/n).read_text()) for n in ('ppo_ground_contact_v2c.py','ppo_ground_contact_v2d.py')]
    methods=[{f.name:ast.dump(f,include_attributes=False) for cls in t.body if isinstance(cls,ast.ClassDef) for f in cls.body if isinstance(f,ast.FunctionDef)} for t in trees]
    for n in ('_matrix','step','update','_complete','reward','reset','close','_attached'):assert methods[0][n]==methods[1][n]
    funcs=[{f.name:ast.dump(f,include_attributes=False) for f in t.body if isinstance(f,ast.FunctionDef)} for t in trees]
    for n in ('configure_sensors','clock','_restore'):assert funcs[0][n]==funcs[1][n]


def test_D_identity_and_all_ancestor_audit_guards():
    src=(ROOT/'scripts/ppo_ground_contact_v2d.py').read_text()
    assert 'ground_contact_reward_v2d as definition' in src and 'GroundContactV2D' in src
    for s in ('v2','v2b','v2c'):assert 'not hasattr(env, "_tron1_ground_contact_'+s+'")' in src
    for s in ('v2a','v2b','v2c'):assert 'immutable_'+s+'_definition_sha256' in src
