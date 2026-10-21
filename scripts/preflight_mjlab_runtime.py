"""Asset-free mjlab import, CUDA-kernel, and physical-contact preflight.

No packages are installed or upgraded.  The explicit source directory allows
using a pinned mjlab checkout with the already-installed RSL-RL 3 runner rather
than silently replacing the learning algorithm with mjlab's newer default.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


PACKAGES = ("torch", "warp-lang", "mujoco", "mujoco-warp", "rsl-rl-lib",
            "tensordict", "numpy", "tyro", "mediapy", "viser")


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def checkout_commit(source):
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_preflight(source, device="cuda:0", steps=64):
    if steps < 40:
        raise ValueError("At least 40 physical steps are needed to test contact")
    source = Path(source).resolve()
    src_path = source / "src"
    if not (src_path / "mjlab" / "__init__.py").is_file():
        raise FileNotFoundError(f"Not a mjlab source checkout: {source}")
    sys.path.insert(0, str(src_path))
    # This global is required by Warp's source compiler, which resolves kernel
    # symbols through the enclosing Python module's globals.
    global wp
    wp = importlib.import_module("warp")
    import numpy as np
    import torch
    import mujoco
    from mjlab.sim import SimulationCfg, MujocoCfg
    from mjlab.sim.sim import Simulation
    # Import full manager-based environment core, not just a raw MJWarp helper.
    from mjlab.envs import ManagerBasedRlEnv

    wp.init()
    result = {
        "mjlab_commit": checkout_commit(source),
        "mjlab_source": str(source), "python": sys.version,
        "packages": package_versions(), "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_driver_version": wp.get_cuda_driver_version(),
        "environment_class": ManagerBasedRlEnv.__module__ + "." + ManagerBasedRlEnv.__name__,
        "notes": ["No assets, training, package installs, or driver changes",
                  "RSL-RL is measured only; upstream mjlab runner is not imported"],
    }

    @wp.kernel
    def increment(values: wp.array(dtype=wp.float32)):
        i = wp.tid()
        values[i] += 1.0

    start = time.monotonic()
    values = wp.zeros(16, dtype=wp.float32, device=device)
    wp.launch(increment, dim=16, inputs=[values], device=device)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(values.numpy(), np.ones(16, dtype=np.float32))
    result["warp_kernel_passed"] = True
    result["warp_kernel_seconds"] = time.monotonic() - start

    xml = """<mujoco model="mjlab-runtime-preflight">
      <option timestep="0.005" gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="ground" type="plane" size="2 2 0.1" friction="0.7 0.005 0.0001"/>
        <body name="ball" pos="0 0 0.16"><freejoint/>
          <geom name="ball_geom" type="sphere" size="0.05" mass="1" friction="0.7 0.005 0.0001"/>
        </body>
      </worldbody>
    </mujoco>"""
    result["model_xml_sha256"] = hashlib.sha256(xml.encode()).hexdigest()
    model = mujoco.MjModel.from_xml_string(xml)
    cfg = SimulationCfg(nconmax=64, njmax=256,
                        mujoco=MujocoCfg(timestep=.005, iterations=10, ls_iterations=5))
    start = time.monotonic()
    sim = Simulation(num_envs=1, cfg=cfg, model=model, device=device)
    initial_z = float(sim.data.qpos[0, 2].item())
    max_contacts = 0
    for _ in range(steps):
        sim.step()
        # nacon is GPU-resident and synchronizing here is deliberate: this is a
        # correctness preflight, not a throughput benchmark.
        max_contacts = max(max_contacts, int(np.max(sim.wp_data.nacon.numpy())))
    wp.synchronize_device(device)
    qpos = sim.data.qpos.detach().cpu().numpy()
    qvel = sim.data.qvel.detach().cpu().numpy()
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise AssertionError("Nonfinite physical state")
    if max_contacts < 1:
        raise AssertionError("The falling sphere never contacted the plane")
    if not (.03 < float(qpos[0, 2]) < .08 < initial_z):
        raise AssertionError(f"Unexpected settled sphere height: {qpos[0, 2]}")
    result["simulation"] = {
        "num_envs": 1, "physics_steps": steps, "dt": .005,
        "initial_z_m": initial_z, "final_z_m": float(qpos[0, 2]),
        "max_contacts": max_contacts, "qpos": qpos.tolist(),
        "qvel": qvel.tolist(), "time_s": sim.data.time.detach().cpu().tolist(),
        "cuda_graphs_enabled": sim.use_cuda_graph,
        "seconds_including_compile": time.monotonic() - start,
        "passed": True,
    }
    # Record actual GPU tensor availability independently of Warp interop.
    result["torch_device"] = str(torch.ones(1, device=device).device)
    result["passed"] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjlab-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_preflight(args.mjlab_source, args.device, args.steps)
    except Exception as error:
        result = {"passed": False, "error_type": type(error).__name__,
                  "error": str(error), "traceback": traceback.format_exc(),
                  "packages": package_versions(),
                  "mjlab_commit": checkout_commit(args.mjlab_source)}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Generated diagnostic output, never a source file or existing result.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
