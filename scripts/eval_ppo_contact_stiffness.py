"""Isolated nominal-friction leg-stiffness test of the frozen action-rate -0.5 PPO.

Reuse the immutable evaluator and provenance guards without enabling its blocked
legacy gain options. This zero-shot deployment diagnostic does not retrain PPO.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys


BASE_ADAPTER_SHA256 = "5353f849d7d4f263c735d2cb000c8000403bf81280b51634065c9b721bb3b277"
BASE_ADAPTER_PATH = Path(__file__).resolve().with_name("eval_ppo_contact_damping.py")


def load_base_adapter(path=BASE_ADAPTER_PATH):
    path = Path(path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != BASE_ADAPTER_SHA256:
        raise ValueError("Frozen damping provenance adapter SHA mismatch")
    spec = importlib.util.spec_from_file_location("_frozen_contact_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_adapter()


def prepare_plan(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--leg-kp", type=float, choices=(500., 250.), required=True)
    options, inherited = parser.parse_known_args(list(argv))
    for word in inherited:
        base.require(not (word == "--leg-kd" or word.startswith("--leg-kd=")),
                     "A stiffness-only test cannot accept --leg-kd")
    # The frozen adapter applies all source, profile, actor, material, budget,
    # fresh-output, sampling and original-flag guards before any app starts.
    base.require(base.sha256(BASE_ADAPTER_PATH) == BASE_ADAPTER_SHA256,
                 "Frozen damping provenance adapter changed")
    plan = base.prepare_plan(["--leg-kd", "10", *inherited])
    plan.pop("leg_kd")
    plan["leg_kp"] = options.leg_kp
    audit = plan["audit"]
    for key in ("deployment_leg_kd", "trained_leg_kd", "unchanged_leg_kp"):
        audit.pop(key)
    audit.update({
        "kind": "fixed_ppo_nominal_contact_stiffness_diagnostic",
        "trained_leg_kp": 500., "deployment_leg_kp": options.leg_kp,
        "unchanged_leg_kd": 10.,
        "only_changed_control_field": "leg_kp" if options.leg_kp != 500. else None,
        "wrapper_sha256": base.sha256(__file__),
        "provenance_adapter_sha256": BASE_ADAPTER_SHA256,
        "interpretation": "One fixed actor with a zero-shot deployment stiffness change; not training at the new gain, a causal proof, or hardware/contact-stability certification.",
    })
    return plan


def apply_config(cfg, evaluation_profile, plan):
    """Only stiffness changes; original damping and every other config stay held."""
    legs = cfg.scene.robot.actuators["legs"]
    base.require(legs.stiffness == 500. and legs.damping == 10., "Unexpected configured original PD")
    base.require(cfg.sim.dt == .005 and cfg.decimation == 4, "Unexpected physical/policy clock")
    base.require(evaluation_profile["contact_friction"] == .7, "Expected nominal evaluation material")
    legs.stiffness = plan["leg_kp"]
    if plan["leg_kp"] != 500.:
        evaluation_profile["leg_kp"] = plan["leg_kp"]


def decorate_contract(contract, plan):
    base.require(contract["leg_kp"] == 500. and contract["leg_kd"] == 10., "Unexpected exported original PD")
    contract["leg_kp"] = plan["leg_kp"]
    contract["contact_stability_diagnostic"] = copy.deepcopy(plan["audit"])


def run(argv):
    plan = prepare_plan(argv)
    source = base.materialize_source(plan["legacy_evaluator"].read_text(), plan["legacy_evaluator"].parent.parent)
    plan["audit"]["generated_evaluator_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    namespace = {"__name__": "_pinned_contact_stiffness_evaluator", "__file__": str(Path(__file__).resolve()),
                 "_contact_plan": plan, "_contact_apply_config": apply_config,
                 "_contact_decorate_contract": decorate_contract}
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(source, "<pinned_contact_stiffness_evaluator>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    run(sys.argv[1:])
