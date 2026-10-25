"""Retain immutable K physical lifecycle;bind only matched S/T arithmetic."""
from __future__ import annotations
import hashlib
from pathlib import Path

PARENT_SHA="3fb2ce270ed5783101bc558cc615e311dd9f586c3096a28184e9174e3d595f2d"


def materialize(source,variant):
    assert variant in ("S","T") and hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA
    letter=variant.lower()
    for old,new in [
        ("import ground_contact_reward_v2k as definition","import ground_contact_reward_v2"+letter+" as definition"),
        ("GroundContactV2K","GroundContactV2"+variant),
        ("_tron1_ground_contact_v2k","_tron1_ground_contact_v2"+letter),
        ("ground_contact_v2k_any_wheel","ground_contact_v2"+letter+"_any_wheel"),
        ('            "support_gap_aggregation": "max8",','            "support_gap_aggregation": self.settings["support_gap_aggregation"],'),
        ('        require(self.num_envs > 0 and not hasattr(env, MARKER)',
         '        require(not any(hasattr(env,m) or hasattr(env.scene,m) for m in ("_tron1_ground_contact_v2k","_tron1_ground_contact_v2s","_tron1_ground_contact_v2t")),"Do not stack K/S/T collectors")\n'
         '        require(self.num_envs > 0 and not hasattr(env, MARKER)'),
        ('        result = {"reward_definition_version":',
         '        result = {"materialized_owner_sha256":_DURATION_MATERIALIZED_SHA256,\n'
         '            "immutable_physical_lifecycle_sha256":"'+PARENT_SHA+'",\n'
         '            "raw_contact_reward_bounds":[-2.,1.],"matched_scale_duration_variant":"'+variant+'",\n'
         '            "reward_definition_version":'),
    ]:
        assert old in source,old;source=source.replace(old,new)
    compile(source,"<matched_duration_owner_"+variant+">","exec");return source


def namespace(variant,path):
    source=materialize((Path(__file__).parent/"ppo_ground_contact_v2k.py").read_text(),variant)
    ns=dict(__name__="_matched_duration_owner_"+variant,__file__=str(Path(path).resolve()),
            _DURATION_MATERIALIZED_SHA256=hashlib.sha256(source.encode()).hexdigest())
    exec(compile(source,"<matched_duration_owner_"+variant+">","exec"),ns)
    return ns
