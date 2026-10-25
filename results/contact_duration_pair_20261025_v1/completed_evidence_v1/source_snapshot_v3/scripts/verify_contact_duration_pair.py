"""Strict matched S/T warm/export/lifecycle/reference verification."""
from __future__ import annotations
import importlib
from pathlib import Path

import verify_contact_camber_reference as parent
import camber_reference_contract as reference


def warm_source(source):
    assert reference.sha(Path(__file__).with_name("verify_contact_warm_refinement.py"))=="8db079293c5f752166d01fb71c0170c32e4db0e6649c151e6f8ebf1c25e226f1"
    for old,new in [("variant in ['K','M']","variant in ['S','T']"),
            ("==(1 if variant=='K' else 4)","==protocol['variant_contact_settings'][variant]['support_gap_weight']")]:
        assert source.count(old)==1,old;source=source.replace(old,new,1)
    return source


def verify_shape(result):
    reference.verify_result(result)
    variant=result["variant"];assert variant in ["S","T"]
    definition=importlib.import_module("ground_contact_reward_v2"+variant.lower())
    settings=definition.validate_settings(result["protocol"]["variant_contact_settings"][variant])
    assert settings==result["contract"]["ground_contact_reward_v2"]
    kp=Path(__file__).resolve().parents[1]/"config/ppo_ground_contact_v2k_400hz_v1.json"
    assert reference.sha(kp)=="855ca5a08e2b228c5146c65fb76f55cf6c9152d70480dc43a32a875aa4ff17d9"
    original=reference.read(kp)
    for key in ["nominal_acceptance_before_training","training_randomization"]:
        assert result["protocol"][key]==original[key]
    for name in ["live_startup","live_final"]:
        a=result["manifest"]["cohort"][name]["contact"]
        assert a["matched_scale_duration_variant"]==variant and a["raw_contact_reward_bounds"]==[-2.,1.]
        assert a["immutable_physical_lifecycle_sha256"]=="3fb2ce270ed5783101bc558cc615e311dd9f586c3096a28184e9174e3d595f2d"
        assert a["support_gap_aggregation"]==settings["support_gap_aggregation"]
        assert a["definition_sha256"]==reference.sha(definition.__file__)
    result["matched_scale_duration_verified"]=True
    return result


def materialize(source):
    assert reference.sha(parent.__file__)=="681d59ce7cecd0796c1e9caae68a6f1956ad46702b2aa181d6bd05155755a7ac"
    source=parent.materialize(source)
    for old,new in [("import train_contact_camber_reference as training","import train_contact_duration_pair as training"),
            ('source = source_path.read_text()','source = _duration_warm_source(source_path.read_text())'),
            ('assert p["variants"] == ["K"] and result["variant"] == "K"','assert p["variants"] == ["S","T"] and result["variant"] in ["S","T"]'),
            ('variant="K", run_kind=','variant=c["variant"], run_kind='),
            ('strict_contact_camber_reference_training_verification','strict_contact_duration_pair_training_verification'),
            ('independent_camber_reference_verified=True, from_scratch=False','independent_camber_reference_verified=True, matched_scale_duration_verified=True, from_scratch=False')]:
        assert old in source,old;source=source.replace(old,new)
    compile(source,"<paired_duration_strict_verifier>","exec");return source


def namespace():
    ns=dict(__name__="_paired_duration_strict_verifier",__file__=str(Path(__file__).resolve()),
        _camber_verify_result=verify_shape,_duration_warm_source=warm_source)
    exec(compile(materialize((Path(__file__).parent/"verify_contact_taskspace_refinement.py").read_text()),"<paired_duration_strict_verifier>","exec"),ns)
    return ns


def verify_training(*args,**kwargs):return namespace()["verify_training"](*args,**kwargs)


if __name__=="__main__":namespace()["main"]()
