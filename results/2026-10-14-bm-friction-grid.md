# Car-drift policy with BeyondMimic-style training randomization

## Setup

This is a **new** TRON1 WF PPO policy, not the earlier fixed-`µ=0.15` baseline. It tracks the locked high-wheel-heading-separation `drift_drl/traj_4` car reference. The retargeted wheel-heading separation is a kinematic reference feature, not a measured tire contact-slip angle. The training reference, task, policy action/observation contract and source hashes are in [training_manifest.json](car_bm_dr_seed42/training_manifest.json); the completed run and live material audit are in [training_status.json](car_bm_dr_seed42/training_status.json).

Training ran on the remote RTX 4090 D with seed 42, 512 parallel environments, 1000 PPO iterations, a 50 Hz actor and 200 Hz physics. The official BeyondMimic `EventCfg` material ranges were mapped to TRON1: startup robot-body static friction U[0.3, 1.6], dynamic friction U[0.3, 1.2], restitution U[0, 0.5], 64 material buckets; joint-default offsets, base CoM perturbations, and 1–3 s velocity pushes were included. The ground material remained at `µ=1.0` during training. The six leg joints receive position offsets; TRON1's two velocity-controlled wheel joints do not. Final live PhysX material readback spanned 0.321–1.575 static, 0.353–1.160 dynamic and 0.001–0.500 restitution over 512 environments × 9 robot bodies. This is a mapping of BeyondMimic's training events to TRON1, not an identical G1 task, robot, or tire model.

The source policy contract retains an inherited `domain_randomization: "disabled for initial pilot"` label from the older TRON1 implementation. For this run, the operative `training_domain_randomization` and top-level `recipe` fields in the manifest, the trainer's installed event functions, and the live material audit document the actual enabled randomization; the inherited label should not be interpreted as the event state.

For validation, the final frozen checkpoint (SHA-256 `f803a8507752d5767a756a537bc844589523de0fb827e669865ac774343db20e`) was reused for every case, with randomization disabled. Robot and ground static/dynamic friction were set to the same fixed value and read back in Isaac/PhysX and MuJoCo. There was one deterministic seed-42 episode per engine and coefficient. The turn core is 1.44–4.00 s; exploratory large-XY screens are reference-core RMSE >0.15 m or peak >0.30 m. Neither is a physical drift-success criterion.

## Top-down XY result

![Reference, Isaac and MuJoCo XY trajectories for six fixed validation friction coefficients](car_bm_dr_seed42/topdown.png)

| Validation µ | Isaac reference-core XY RMSE | MuJoCo reference-core XY RMSE | Isaac–MuJoCo core XY RMSE | Full turn core? | Large-XY screen? |
| ---: | ---: | ---: | ---: | :---: | :---: |
| 0.30 | 0.0436 m | 0.0323 m | 0.0212 m | Both | No |
| 0.45 | 0.0335 m | 0.0340 m | 0.0169 m | Both | No |
| 0.60 | 0.0340 m | 0.0354 m | 0.0142 m | Both | No |
| 0.70 | 0.0253 m | 0.0372 m | 0.0249 m | Both | No |
| 0.80 | 0.0247 m | 0.0303 m | 0.0138 m | Both | No |
| 1.00 | 0.0336 m | 0.0426 m | 0.0272 m | Both | No |

All twelve rollouts reached the 6.98 s motion end without an early stop. The largest core reference-XY peak was 0.126 m (Isaac, µ=0.30), below the exploratory 0.30 m screen. Within this **tested equal-µ grid**, the BeyondMimic-randomized policy did not show a large root-XY deviation. No friction failure boundary has been established: values above 1.0, unequal static/dynamic pairs, lower-friction cases and multi-seed statistics remain untested here. Equal numerical friction coefficients do not imply calibrated equal tire–ground physics across PhysX and MuJoCo. Root XY agreement alone does not establish actual wheel-patch sliding or human GMR retarget feasibility.

Compact [native-clock XY traces](car_bm_dr_seed42/xy_trace.npz), [all metrics](car_bm_dr_seed42/summary.json), and [case-by-case readback and SHA-256 provenance](car_bm_dr_seed42/xy_trace.provenance.json) are published; licensed robot assets and third-party raw car/mocap data are not. The published figure and metrics can be independently regenerated without Isaac or MuJoCo:

```bash
python scripts/analyze_friction_grid.py \
  --trace results/car_bm_dr_seed42/xy_trace.npz \
  --expected-summary results/car_bm_dr_seed42/summary.json \
  --output-dir outputs/replayed_bm_grid
```

The full training and paired physics rerun commands are in the top-level README. The training checkpoint remains on the authorized remote host at `/data/zengxiong/workspaces/tron1-sliding-analysis-20261014/train_bm_dr_seed42/model_final.pt`; it is not redistributed here.
