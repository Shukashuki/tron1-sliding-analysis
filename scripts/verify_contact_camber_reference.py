"""Strict original PPO warm/export proof plus native new-reference ingestion."""
from __future__ import annotations
from pathlib import Path
import sys

import verify_contact_wheel_axis as parent
import camber_reference_contract as reference


def materialize(source):
    assert reference.sha(parent.__file__)=="04db82e4ae8d45773a329cf354b05895f07ecc0cb1d61cc4e8acde989eb6f896"
    source=parent.materialize(source)
    for old,new in [
        ("import train_contact_wheel_axis as training","import train_contact_camber_reference as training"),
        ('    result["three_dimensional_axial_orientation_verified"] = True\n',
         '    result["three_dimensional_axial_orientation_verified"] = True\n    _camber_verify_result(result)\n'),
        ('strict_contact_wheel_axis_training_verification','strict_contact_camber_reference_training_verification'),
        ('three_dimensional_axial_orientation_verified=True, from_scratch=False',
         'three_dimensional_axial_orientation_verified=True, independent_camber_reference_verified=True, from_scratch=False'),
    ]:
        assert source.count(old)>=1,old;source=source.replace(old,new)
    compile(source,"<strict_camber_reference_training_proof>","exec");return source


def _namespace():
    ns=dict(__name__="_strict_camber_reference_training_proof",__file__=str(Path(__file__).resolve()),_camber_verify_result=reference.verify_result)
    exec(compile(materialize((Path(__file__).parent/"verify_contact_taskspace_refinement.py").read_text()),"<strict_camber_reference_training_proof>","exec"),ns)
    return ns


def verify_training(*args,**kwargs):return _namespace()["verify_training"](*args,**kwargs)


if __name__=="__main__":_namespace()["main"]()
