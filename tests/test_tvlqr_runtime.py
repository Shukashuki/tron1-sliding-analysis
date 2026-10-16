import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
pytest.importorskip("mujoco")
from run_tvlqr_tracking import ROOT, feedback_action, load_design, run


def design_arrays():
    return {"nominal_states":np.zeros((350,29)),"nominal_actions":np.zeros((349,8)),
        "nominal_previous_actions":np.zeros((349,8)),"K":np.zeros((349,8,36)),"feed":np.zeros((349,8))}


def test_feedback_uses_actual_state_and_previous_applied_command_not_action_playback():
    design=design_arrays();design["nominal_actions"][4]=.125
    design["K"][4,0,0]=-2.;design["K"][4,1,28]=3.;design["feed"][4,2]=.25
    error=np.zeros(28);error[0]=.1;previous=np.zeros(8);previous[0]=.25
    command,info=feedback_action(design,4,error,previous)
    np.testing.assert_array_equal(command,np.array([-.075,.875,.375,.125,.125,.125,.125,.125]).astype(np.float32).astype(float))
    assert info["action_replay"] is False and info["online_optimization"] is False
    assert info["clipped_action_components"]==0
    assert not np.array_equal(command,design["nominal_actions"][4])


def test_unconstrained_feedback_is_clipped_then_quantized_to_shared_float32_interface():
    design=design_arrays();design["feed"][0]=[4,-2,1,-1,.123456789,0,0,0]
    command,info=feedback_action(design,0,np.zeros(28),np.zeros(8))
    np.testing.assert_array_equal(command,np.clip(design["feed"][0],-1,1).astype(np.float32).astype(float))
    assert info["clipped_action_components"]==2


@pytest.mark.parametrize("frame,error,previous",[
    (True,np.zeros(28),np.zeros(8)),(-1,np.zeros(28),np.zeros(8)),(349,np.zeros(28),np.zeros(8)),
    (0,np.zeros(29),np.zeros(8)),(0,np.full(28,np.nan),np.zeros(8)),
    (0,np.zeros(28),np.full(8,1.01)),(0,np.zeros(28),np.full(8,.1)),
    (0,np.zeros(28),np.zeros(7)),
])
def test_feedback_rejects_invalid_state_action_phase(frame,error,previous):
    with pytest.raises(ValueError):
        feedback_action(design_arrays(),frame,error,previous)


def test_nonfinite_gain_command_propagates_instead_of_hidden_fallback():
    design=design_arrays();design["feed"][0,0]=np.inf
    with pytest.raises(FloatingPointError):
        feedback_action(design,0,np.zeros(28),np.zeros(8))


def test_design_loader_binds_bytes_model_motion_and_actual_action_memory(tmp_path):
    inputs={}
    for name in ("model","motion","contract"):
        inputs[name]=tmp_path/name;inputs[name].write_bytes(name.encode())
    folder=tmp_path/"design";folder.mkdir();arrays=design_arrays()
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def publish():
        np.savez_compressed(folder/"design.npz",**arrays)
        metadata={"kind":"model_nominal_initialized_height_smooth_tvlqr_design","status":"completed",
            "nominal_prediction_mu":.7,"design_sha256":digest(folder/"design.npz"),
            "model_sha256":digest(inputs["model"]),"motion_sha256":digest(inputs["motion"]),
            "actuator_contract_sha256":digest(inputs["contract"]),
            "source_sha256":{str((ROOT/relative).resolve()):digest(ROOT/relative) for relative in (
                "scripts/classical_tracking.py","scripts/run_nmpc_tracking.py",
                "upstream/tron1-learning-from-human/scripts/eval_tracking_mujoco.py")}}
        (folder/"report.json").write_text(json.dumps(metadata))
    publish();report,actual=load_design(folder,**inputs)
    np.testing.assert_array_equal(actual["K"],arrays["K"])
    assert report["nominal_prediction_mu"]==.7
    invalid=json.loads((folder/"report.json").read_text());invalid["source_sha256"]={}
    (folder/"report.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError,match="source identity"):
        load_design(folder,**inputs)
    publish()
    arrays["nominal_previous_actions"][10,0]=.25;publish()
    with pytest.raises(ValueError,match="phase"):
        load_design(folder,**inputs)
    arrays["nominal_previous_actions"][:]=0;arrays["nominal_actions"][0,0]=.1;publish()
    with pytest.raises(ValueError,match="float32"):
        load_design(folder,**inputs)
    arrays["nominal_actions"][:]=0;publish();inputs["motion"].write_bytes(b"changed")
    with pytest.raises(ValueError,match="identity"):
        load_design(folder,**inputs)


def test_preserve_existing_episode_before_reading_model(tmp_path):
    from types import SimpleNamespace
    with pytest.raises(FileExistsError):
        run(SimpleNamespace(output_dir=tmp_path))
