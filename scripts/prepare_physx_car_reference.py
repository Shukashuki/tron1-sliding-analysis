"""Export unscaled traj_4 states/controls for a labeled PhysX car surrogate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def run(a):
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/"upstream/tron1-learning-from-human/scripts"))
    import prepare_drift_drl_turn_reference as source
    config_path=root/"upstream/tron1-learning-from-human/config/drift_turn_source.json"
    cfg=json.loads(config_path.read_text());csv=root/"upstream/drift_drl/code/ref_trajectory/traj_4.csv"
    assert hashlib.sha256(csv.read_bytes()).hexdigest()==cfg["sha256"]
    full=source.decode_csv(csv.read_bytes());selected,evidence=source.selected_source(full,cfg)
    ids=full["unique_pose_indices"]
    ids=ids[(ids>=cfg["source_row_start_inclusive"])&(ids<=cfg["source_row_end_inclusive"])]
    t=full["time_s_estimated"][ids]-full["time_s_estimated"][ids[0]]
    yaw0=full["yaw_rad"][ids[0]]
    R=np.array([[np.cos(yaw0),np.sin(yaw0)],[-np.sin(yaw0),np.cos(yaw0)]])
    xy=(full["position_xy_m"][ids]-full["position_xy_m"][ids[0]])@R.T
    velocity=full["velocity_xy_m_s"][ids]@R.T
    yaw=np.unwrap(full["yaw_rad"][ids])-yaw0
    controls=full["controls"][ids].copy();controls[:,0]*=-1 # CARLA right-positive to right-handed left-positive
    assert np.all(np.diff(t)>0) and a.output.suffix==".npz" and not a.output.exists()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,time_s_estimated=t,position_xy_m=xy,velocity_xy_m_s=velocity,
        yaw_rad=yaw,beta_rad=full["beta_rad"][ids],normalized_steer_throttle_brake_handbrake=controls,
        source_row=ids,core_time_bounds_s_estimated=full["time_s_estimated"][[cfg["drift_row_start_inclusive"],cfg["drift_row_end_inclusive"]]]-full["time_s_estimated"][ids[0]])
    proof={"source_csv_sha256":cfg["sha256"],"source_decoder_sha256":hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        "exporter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256":hashlib.sha256(a.output.read_bytes()).hexdigest(),"evidence":evidence,
        "spatial_scale":1.,"time_scale":1.,"clock_is_estimated":True,"per_wheel_observations_available":False,
        "normalized_steering_is_not_measured_road_wheel_angle":True,
        "exact_original_car_plant_identified":False,"physics_task_result":False}
    a.output.with_suffix(".json").write_text(json.dumps(proof,indent=2)+chr(10));print(json.dumps(proof))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
