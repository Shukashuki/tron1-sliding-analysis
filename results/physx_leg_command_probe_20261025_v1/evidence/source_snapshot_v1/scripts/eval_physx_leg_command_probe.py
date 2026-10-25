"""Frozen original-K actor: exact passthrough versus physical leg-target ramp.

Counted hooks retain the audited original single-cylinder PhysX evaluator.
The intervention is explicitly an action-execution change; its records cannot
enter the unchanged-action wheel-model comparison or certify a trained model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import physx_leg_command_probe as probe

PARENT_SHA = "32b690dac706929b4df582a1b758fb0cbebe242f9ca8c5646f6c064d2a2eec62"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    replacements = [
        ('        runtime.append(tire.TireRuntime(env,law=a.law,mu=.7))\n',
         '        runtime.append(tire.TireRuntime(env,law=a.law,mu=.7))\n        _probe_install(env)\n'),
        ('        report["wheel_contact_model_transfer"]=metadata;contract["wheel_contact_model_transfer"]=metadata\n',
         '        _probe_finish(metadata, report, contract, current, observed, proof)\n'
         '        report["wheel_contact_model_transfer"]=metadata;contract["wheel_contact_model_transfer"]=metadata\n')]
    for old, new in replacements:
        assert source.count(old) == 1, "Missing/ambiguous command probe hook"
        source = source.replace(old, new, 1)
    compile(source, "<K_command_probe_parent>", "exec")
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--contact-model-evaluator", type=Path, required=True)
    parser.add_argument("--command-mode", choices=probe.MODES, required=True)
    parser.add_argument("--probe-protocol", type=Path, required=True)
    own, remaining = parser.parse_known_args(argv)
    protocol = json.loads(own.probe_protocol.read_text())
    assert protocol["kind"] == "frozen_K_PhysX_native_leg_target_execution_diagnostic"
    assert protocol["modes"] == list(probe.MODES) and protocol["mu"] == .7
    assert protocol["physics_dt_s"] == .0025 and protocol["policy_dt_s"] == .02
    assert protocol["physical_steps"] == 2792 and protocol["policy_decisions"] == 349
    for flag, value in (("--geometry", "authored"), ("--law", "native_coulomb"), ("--contact-offset-m", ".001"), ("--rest-offset-m", "0")):
        assert remaining.count(flag) == 1 and remaining[remaining.index(flag)+1] == value
    assert sha(own.contact_model_evaluator) == PARENT_SHA
    source = materialize(own.contact_model_evaluator.read_text())
    source_proof = {str(Path(path).resolve()): sha(path) for path in (__file__, probe.__file__, own.contact_model_evaluator, own.probe_protocol)}
    owners = []

    def install(env):
        assert not owners
        owners.append(probe.LegCommandProbe(env, own.command_mode))

    def finish(metadata, report, contract, plan, recording_sha, baseline_proof):
        assert len(owners) == 1 and all(sha(path) == digest for path,digest in source_proof.items())
        output = plan["output"]
        command = owners[0].save(output/"leg_command_trajectory.npz")
        if own.command_mode == "passthrough":
            assert recording_sha == baseline_proof["recording_sha256"], "Read-only command probe changed the native recording"
            assert recording_sha == protocol["baseline_physics_trajectory_sha256"]
        assert metadata["normalized_actor_sha256"] == protocol["actor_sha256"]
        metadata["kind"] = "K_policy_PhysX_leg_command_execution_diagnostic"
        metadata["retraining_or_reward_action_reference_change"] = own.command_mode != "passthrough"
        metadata["command_execution_diagnostic"] = command
        metadata["leg_command_trajectory_sha256"] = sha(output/"leg_command_trajectory.npz")
        metadata["source_sha256"].update(source_proof)
        metadata["additional_training_updates"] = 0
        metadata["command_probe_protocol_sha256"] = sha(own.probe_protocol)
        contract["action_execution_intervention"] = command
        report["action_execution_intervention"] = command
        with (output/"leg_command_status.json").open("x") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)

    namespace = dict(__name__="_K_command_execution_probe", __file__=str(own.contact_model_evaluator.resolve()),
        _probe_install=install, _probe_finish=finish)
    previous = sys.argv
    try:
        sys.path.append(str(own.contact_model_evaluator.resolve().parent))
        sys.argv = [str(own.contact_model_evaluator), *remaining]
        exec(compile(source, "<K_command_probe_parent>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main(sys.argv[1:])
