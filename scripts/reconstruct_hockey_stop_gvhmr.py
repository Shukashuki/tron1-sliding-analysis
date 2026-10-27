#!/usr/bin/env python3
"""Run pinned GVHMR on CPU, preserving raw sliding before foot-lock correction.

Uses upstream YOLO track selection/smoothing, ViTPose flip testing, HMR2 and
SimpleVO. Only device/batch handling and the explicit postproc=False primary
output differ from the upstream demo. Input must be 30 fps: GVHMR was trained
with that temporal convention. This is estimated monocular world motion, not
a calibrated measurement of speed, contact, blade orientation or forces.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GVHMR = ROOT / "upstream/GVHMR"
COMMIT = "ee960bb6e2ea2d381aa97f08e9b71ef320b624b1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--body-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reuse-preprocess", type=Path, help="Explicit cache reuse from a matching source/checkpoint run")
    parser.add_argument("--camera-rotations", type=Path, help="Optional audited approximate camera rotation artifact")
    parser.add_argument("--camera-report", type=Path, help="Required provenance for --camera-rotations")
    args = parser.parse_args()
    if bool(args.camera_rotations) != bool(args.camera_report):
        raise ValueError("Supply both camera rotations and their report")
    args.video = args.video.resolve()
    args.checkpoints = args.checkpoints.resolve()
    args.body_models = args.body_models.resolve()
    args.output = args.output.resolve()
    if min(args.threads, args.batch_size) < 1:
        raise ValueError("Threads and batch size must be positive")
    if subprocess.check_output(["git", "-C", str(GVHMR), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise ValueError("GVHMR commit differs from experiment pin")
    # The official model factory resolves its body-model path against PROJ_ROOT.
    link = GVHMR / "inputs/checkpoints/body_models"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve() != args.body_models:
            raise ValueError(f"Refusing to replace existing body-model path: {link}")
    else:
        link.symlink_to(args.body_models, target_is_directory=True)
    sys.path.insert(0, str(GVHMR))

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from ultralytics import YOLO
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # registers demo Hydra model
    from hmr4d.utils.preproc.tracker import Tracker
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch
    from hmr4d.utils.preproc.vitpose_pytorch import build_model
    from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO
    from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps
    from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17
    from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy, normalize_kp2d
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.network.hmr2 import load_hmr2
    from hmr4d.utils.body_model.smplx_lite import SmplxLite, batch_rigid_transform_v2
    from pytorch3d.transforms import axis_angle_to_matrix

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if abs(fps - 30) > 0.01 or count < 2:
        raise ValueError("Provide a full 30-fps clip; do not silently change temporal scale")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "video_sha256": sha(args.video), "video": str(args.video),
        "gvhmr_commit": COMMIT, "runner_sha256": sha(__file__),
        "fps": fps, "frame_count": count, "device": "cpu",
        "batch_size": args.batch_size, "threads": args.threads,
        "metric_world_trajectory_verified": False,
        "scale_status": "monocular model estimate; no external metric calibration",
        "contact_forces_observed": False, "retarget_performed": False,
        "primary_postproc": False, "static_cam": False,
        "checkpoint_inventory": json.loads((args.checkpoints / "download_inventory.json").read_text()),
        "body_model_sha256": sha(args.body_models / "smplx/SMPLX_NEUTRAL.npz"),
    }
    from download_hockey_gvhmr_weights import WEIGHTS
    for name, (size, expected) in WEIGHTS.items():
        checkpoint = args.checkpoints / name
        if checkpoint.stat().st_size != size or sha(checkpoint) != expected:
            raise ValueError(f"Checkpoint does not match pinned SHA256: {name}")
    if args.reuse_preprocess:
        old = json.loads((args.reuse_preprocess / "provenance.json").read_text())
        for key in ("video_sha256", "gvhmr_commit", "checkpoint_inventory", "frame_count"):
            if old[key] != manifest[key]:
                raise ValueError(f"Preprocessing cache differs: {key}")
        manifest["preprocessing_cache"] = {"source": str(args.reuse_preprocess),
            "source_runner_sha256": old["runner_sha256"], "files": {}}
        for name in ("bbx.pt", "vitpose.pt", "vit_features.pt"):
            original = args.reuse_preprocess / name
            target = args.output / name
            if target.exists() and sha(target) != sha(original):
                raise ValueError(f"Existing cache differs: {name}")
            if not target.exists():
                shutil.copy2(original, target)
            manifest["preprocessing_cache"]["files"][name] = sha(target)
    manifest_path = args.output / "provenance.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ("video_sha256", "gvhmr_commit", "runner_sha256"):
            if old[key] != manifest[key]:
                raise ValueError(f"Cache provenance differs: {key}. Use a new output directory.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.monotonic()

    class CPUTracker(Tracker):
        def __init__(self):
            self.yolo = YOLO(args.checkpoints / "yolo/yolov8x.pt")

        def track(self, video_path):
            history = []
            results = self.yolo.track(video_path, device="cpu", conf=0.5,
                                      classes=0, verbose=False, stream=True)
            for result in tqdm(results, total=count, desc="CPU YOLO tracking"):
                frame = []
                if result.boxes.id is not None:
                    ids = result.boxes.id.int().cpu().tolist()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    frame = [{"id": ids[i], "bbx_xyxy": boxes[i]} for i in range(len(ids))]
                history.append(frame)
            return history

    with torch.inference_mode():
        path = args.output / "bbx.pt"
        if not path.exists():
            tracker = CPUTracker()
            boxes = tracker.get_one_track(str(args.video)).float()
            torch.save({"bbx_xyxy": boxes, "bbx_xys": get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2).float()}, path)
            del tracker
            gc.collect()
        boxes = torch.load(path, map_location="cpu")["bbx_xys"]
        imgs = None
        pose_path = args.output / "vitpose.pt"
        feat_path = args.output / "vit_features.pt"
        if not pose_path.exists() or not feat_path.exists():
            imgs, _ = get_batch(str(args.video), boxes, img_ds=0.5)

        if not pose_path.exists():
            pose = build_model("ViTPose_huge_coco_256x192").cpu().eval()
            state = torch.load(args.checkpoints / "vitpose/vitpose-h-multi-coco.pth", map_location="cpu")
            pose.load_state_dict(state["state_dict"], strict=True)
            del state
            keypoints = []
            for j in tqdm(range(0, count, args.batch_size), desc="CPU ViTPose (flip test)"):
                batch = imgs[j:j + args.batch_size, :, :, 32:224]
                heatmap, flipped = pose(torch.cat([batch, batch.flip(3)], dim=0)).chunk(2)
                heatmap = (heatmap + flip_heatmap_coco17(flipped)) * 0.5
                bb = boxes[j:j + args.batch_size]
                scale = (torch.cat((bb[:, [2]] * 24 / 32, bb[:, [2]]), dim=1) / 200).numpy()
                pred, conf = keypoints_from_heatmaps(heatmaps=heatmap.numpy(), center=bb[:, :2].numpy(), scale=scale, use_udp=True)
                keypoints.append(torch.from_numpy(np.concatenate((pred, conf), axis=-1)))
            torch.save(torch.cat(keypoints), pose_path)
            del pose, keypoints, heatmap, flipped
            gc.collect()

        if not feat_path.exists():
            model = load_hmr2(args.checkpoints / "hmr2/epoch=10-step=25000.ckpt").cpu().eval()
            features = []
            for j in tqdm(range(0, count, args.batch_size), desc="CPU HMR2 features"):
                features.append(model({"img": imgs[j:j + args.batch_size]}).cpu())
            torch.save(torch.cat(features).clone(), feat_path)
            del model, features
            gc.collect()
        del imgs

        if args.camera_rotations:
            camera_report = json.loads(args.camera_report.read_text())
            if camera_report["video_sha256"] != manifest["video_sha256"] or camera_report["rotations_sha256"] != sha(args.camera_rotations):
                raise ValueError("Camera artifact/source hash differs from report")
            rotations = torch.from_numpy(torch.load(args.camera_rotations, map_location="cpu")).float()
            manifest["camera_method"] = camera_report
        else:
            vo_path = args.output / "simplevo.pt"
            if not vo_path.exists():
                vo = SimpleVO(str(args.video), scale=0.5, step=8, method="sift")
                torch.save(vo.compute(), vo_path)
            rotations = torch.from_numpy(torch.load(vo_path, map_location="cpu")[:, :3, :3]).float()
            manifest["camera_method"] = "upstream SimpleVO SIFT, serial, step=8; external audit still required"
        if rotations.shape != (count, 3, 3) or not torch.isfinite(rotations).all():
            raise ValueError("Invalid camera rotations")
        torch.testing.assert_close(rotations @ rotations.mT, torch.eye(3).expand(count,3,3), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(count), atol=1e-4, rtol=1e-4)
        data = {
            "length": torch.tensor(count), "bbx_xys": boxes,
            "kp2d": torch.load(pose_path, map_location="cpu"),
            "K_fullimg": estimate_K(width, height).repeat(count, 1, 1),
            "cam_angvel": compute_cam_angvel(rotations),
            "f_imgseq": torch.load(feat_path, map_location="cpu"),
        }
        register_store_gvhmr()
        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            cfg = compose(config_name="demo", overrides=[f"video_name={args.video.stem}"])
        model = instantiate(cfg.model, _recursive_=False).cpu().eval()
        model.load_pretrained_model(args.checkpoints / "gvhmr/gvhmr_siga24_release.ckpt")
        batch = {k: v[None] for k, v in data.items() if k != "kp2d"}
        batch["obs"] = normalize_kp2d(data["kp2d"], boxes)[None]
        skeleton = SmplxLite()
        for postproc, label in ((False, "raw"), (True, "standard_postproc")):
            outputs = model.pipeline.forward(batch, train=False, postproc=postproc, static_cam=False)
            params = {k: v[0].cpu().clone() for k, v in outputs["pred_smpl_params_global"].items()}
            result = {"smpl_params_global": params, "K_fullimg": data["K_fullimg"], "postproc": postproc}
            torch.save(result, args.output / f"hmr4d_{label}.pt")
            n = len(params["transl"])
            full_pose = torch.cat([params["global_orient"], params["body_pose"], skeleton.other_default_pose.expand(n, -1)], dim=-1)
            joint_pos, _ = batch_rigid_transform_v2(axis_angle_to_matrix(full_pose.reshape(n, 55, 3)), skeleton.get_skeleton(params["betas"]), skeleton.parents)
            joint_pos = joint_pos + params["transl"][:, None]
            np.savez_compressed(args.output / f"world_{label}.npz", time_s=np.arange(n) / fps,
                                joints_world_estimated_m=joint_pos.numpy(),
                                **{k: v.numpy() for k, v in params.items()},
                                metric_world_trajectory_verified=False, postproc=postproc)
        manifest["elapsed_s"] = time.monotonic() - start
        manifest["completed"] = True
        manifest["outputs"] = {p.name: sha(p) for p in args.output.glob("world_*.npz")}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
