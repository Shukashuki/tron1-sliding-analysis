"""Portable capture IO tests; synthetic bytes/mocked physical audits only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import assemble_sa29_capture as assembly
from test_sa29_policy_friction_dynamics import manifest_fixture


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path, root): return {"path": str(path.relative_to(root)), "sha256": sha(path)}


def write(path, value): path.write_text(json.dumps(value))


def policy_plan(root):
    manifest = manifest_fixture(root)
    source = root / "evaluation_source"
    names, _ = assembly.policy.expected_evaluation_sources()
    hashes = {}
    for name in names:
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((name+" synthetic source token").encode()); hashes[name] = sha(path)
    helper = source / "training/tron1_terrain.py"
    helper.write_bytes((ROOT/"upstream/tron1-learning-from-human/training/tron1_terrain.py").read_bytes())
    ignored = source / "__pycache__/ignored.pyc"; ignored.parent.mkdir(); ignored.write_bytes(b"not an original source binding")
    manifest["evaluation_source"]["source_sha256"] = hashes
    document = {"schema_version": 1, "kind": "sa29_policy_capture_assembly_plan", "reference_id": assembly.policy.REFERENCE_ID,
        "original_root": ".", **{name: copy.deepcopy(manifest[name]) for name in ("inputs", "evaluation_source", "policies", "cases")}}
    path = root / "plan.json"; write(path, document)
    return path, document


def mock_policy_audits(monkeypatch):
    calls = []
    def audit(document, root):
        calls.append(root)
        # Retain frozen complete34/PPO4/SAC6/portable binding parser in the IO test.
        assembly.policy.parse_manifest(document, root)
    monkeypatch.setattr(assembly, "_audit_policy", audit)
    return calls


def test_policy_copies_only_original_whitelist_and_audits_before_atomic_publish(tmp_path, monkeypatch):
    plan, document = policy_plan(tmp_path); calls = mock_policy_audits(monkeypatch)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    output = tmp_path / "portable_capture"
    result = assembly.assemble_policy(plan, output, tmp_path / "future_analysis")
    assert len(calls) == 2 and calls[0] == tmp_path and calls[1] != output
    assert all(p.read_bytes() == value for p, value in before.items())
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["cases"]) == 34 and set(manifest["policies"]["PPO"]) == {"manifest", "status", "checkpoint", "runner_config"}
    assert len(list((output / "evaluation_source").rglob("*.py"))) == 13
    assert len([p for p in (output / "evaluation_source").rglob("*") if p.is_file()]) == 14
    assert not list(output.rglob("*.pyc"))
    original_receipt = tmp_path / document["cases"][0]["execution"]["isaac"]["receipt"]["path"]
    copied_receipt = output / manifest["cases"][0]["execution"]["isaac"]["receipt"]["path"]
    assert original_receipt.read_bytes() == copied_receipt.read_bytes()
    assert result["analysis_executed"] is result["simulation_performed"] is False
    assert not (tmp_path / "future_analysis").exists()


@pytest.mark.parametrize("problem", ["partial", "duplicate", "missing_PPO_file", "extra_PPO_contract", "missing_SAC_contract",
    "missing_receipt", "changed_hash", "absolute", "parent", "symlink", "alias_path", "alias_inode"])
def test_policy_rejects_incomplete_changed_or_nonportable_originals_without_publishing(tmp_path, monkeypatch, problem):
    plan, document = policy_plan(tmp_path); mock_policy_audits(monkeypatch)
    first = document["cases"][0]
    if problem == "partial": document["cases"].pop()
    elif problem == "duplicate": document["cases"][-1] = copy.deepcopy(first)
    elif problem == "missing_PPO_file": del document["policies"]["PPO"]["runner_config"]
    elif problem == "extra_PPO_contract": document["policies"]["PPO"]["policy_contract"] = document["policies"]["SAC"]["policy_contract"]
    elif problem == "missing_SAC_contract": del document["policies"]["SAC"]["policy_contract"]
    elif problem == "missing_receipt": del first["execution"]["isaac"]["receipt"]
    elif problem == "changed_hash": first["files"]["isaac_trace"]["sha256"] = "0" * 64
    elif problem == "absolute": first["files"]["isaac_trace"]["path"] = str(tmp_path / first["files"]["isaac_trace"]["path"])
    elif problem == "parent": first["files"]["isaac_trace"]["path"] = "../outside.npz"
    elif problem == "symlink":
        original = tmp_path / first["files"]["isaac_trace"]["path"]; extra = original.with_name("original-token")
        original.rename(extra); original.symlink_to(extra)
    elif problem == "alias_path": document["cases"][1]["files"]["isaac_trace"] = first["files"]["isaac_trace"]
    else:
        original = tmp_path / first["files"]["isaac_trace"]["path"]
        target = tmp_path / document["cases"][1]["files"]["isaac_trace"]["path"]
        target.unlink(); os.link(original, target)
    write(plan, document)
    with pytest.raises((ValueError, KeyError)):
        assembly.assemble_policy(plan, tmp_path / "must_not_publish", tmp_path / "future")
    assert not (tmp_path / "must_not_publish").exists()


@pytest.mark.parametrize("problem", ["original_audit", "staged_audit", "target_appeared"])
def test_policy_audit_failure_or_concurrent_target_never_overwrites_or_publishes_partial_capture(tmp_path, monkeypatch, problem):
    plan, _ = policy_plan(tmp_path); output = tmp_path / "capture"; calls = []
    def audit(document, root):
        calls.append(root)
        if (problem == "original_audit" and len(calls) == 1) or (problem == "staged_audit" and len(calls) == 2):
            raise ValueError("synthetic scientific audit rejection")
        if problem == "target_appeared" and len(calls) == 2:
            output.mkdir(); (output / "user-evidence").write_text("preserve")
    monkeypatch.setattr(assembly, "_audit_policy", audit)
    with pytest.raises(ValueError): assembly.assemble_policy(plan, output, tmp_path / "future")
    if problem == "target_appeared": assert (output / "user-evidence").read_text() == "preserve"
    else: assert not output.exists()


def nmpc_plan(root):
    def file(relative, value=b"synthetic original token"):
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
        return bind(path, root)
    inputs = {name: file("inputs/" + filename) for name, filename in assembly.INPUT_NAMES.items()}
    for name in assembly.policy.BASE_SOURCE_SHA256:
        path = root / "source" / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / "upstream/tron1-learning-from-human" / name).read_bytes())
    # The frozen bundle verifier is mocked only in IO tests; fresh arbitrary
    # bundles are not accepted by the production assembly path.
    bundle_hashes = {name: file("bundle/" + name)["sha256"] for name in assembly.comparison.MUJOCO_BUNDLE_SHA256}
    cases = []
    for mu in assembly.comparison.GRID:
        prefix = "raw/" + assembly.comparison.case_name(mu)
        files = {role: file(prefix + "/" + name) for role, name in assembly.NMPC_NAMES.items()}
        receipt = file(prefix + "/terminal.json", json.dumps({"capture_complete": True,
            "owned_processes_and_contexts_absent": True, "source_bindings_unchanged": True}).encode())
        cases.append({"mu": mu, "files": files, "receipt": receipt, "log": file(prefix + "/full.log")})
    document = {"schema_version": 1, "kind": "sa29_nmpc_execution_assembly_plan", "reference_id": assembly.policy.REFERENCE_ID,
        "original_root": ".", "inputs": inputs, "source_root": "source", "mujoco_bundle": "bundle", "cases": cases}
    path = root / "nmpc-plan.json"; write(path, document)
    return path, document, bundle_hashes


def mock_nmpc_audits(monkeypatch, root, bundle_hashes):
    calls = []
    monkeypatch.setattr(assembly, "_inputs", lambda document, base: ({name: assembly.original_file(value, base)
        for name, value in document["inputs"].items()}, {}))
    monkeypatch.setattr(assembly, "load_arrays", lambda _: {})
    monkeypatch.setattr(assembly, "reference_trace", lambda _: {})
    monkeypatch.setattr(assembly.comparison, "MUJOCO_BUNDLE_SHA256", bundle_hashes)
    monkeypatch.setattr(assembly.comparison, "load_mujoco_bundle", lambda *unused: ({"runs": {
        assembly.comparison.case_name(mu): {} for mu in assembly.comparison.GRID}}, None, None, None, None, None))
    def audit(directory, mu, spec, model, mj_record, *, capture):
        calls.append((directory, mu))
        return {}, {}, {name: {} for name in ("controller_contract", "native_identity", "asset_sha256", "fk_source_helpers_sha256")}
    monkeypatch.setattr(assembly.comparison, "audit_isaac_episode", audit)
    monkeypatch.setattr(assembly, "validate_case_against_nominal", lambda *unused: None)
    return calls


def test_nmpc_private_tree_is_complete_and_next_command_uses_locked_replay_not_new_bundle_claim(tmp_path, monkeypatch):
    plan, _, hashes = nmpc_plan(tmp_path); calls = mock_nmpc_audits(monkeypatch, tmp_path, hashes)
    result = assembly.assemble_nmpc(plan, tmp_path / "capture", tmp_path / "future")
    assert len(calls) == 34
    manifest = assembly.comparison.load_execution_manifest(tmp_path / "capture/execution_manifest.json")
    assert set(manifest["cases"]) == set(assembly.comparison.GRID)
    assert result["locked_published_mujoco_bundle_required"] is True and result["arbitrary_new_mujoco_bundle_supported"] is False
    argv = result["analysis_command"]
    assert argv.count("--run") == 17 and "--execution-manifest" in argv
    assert result["analysis_executed"] is result["simulation_performed"] is False and not (tmp_path / "future").exists()


def test_nmpc_calls_the_frozen_locked_bundle_verifier_and_preserves_rejection(tmp_path, monkeypatch):
    plan, _, hashes = nmpc_plan(tmp_path); mock_nmpc_audits(monkeypatch, tmp_path, hashes)
    calls = []
    def reject(*args): calls.append(args); raise ValueError("not the immutable published17 bundle")
    monkeypatch.setattr(assembly.comparison, "load_mujoco_bundle", reject)
    with pytest.raises(ValueError, match="immutable published17"):
        assembly.assemble_nmpc(plan, tmp_path / "capture", tmp_path / "future")
    assert len(calls) == 1 and not (tmp_path / "capture").exists()


@pytest.mark.parametrize("problem", ["partial", "duplicate", "source_changed", "bad_exit_capture", "artifact_hash", "alias"])
def test_nmpc_incomplete_or_changed_evidence_is_never_promoted(tmp_path, monkeypatch, problem):
    plan, document, hashes = nmpc_plan(tmp_path); mock_nmpc_audits(monkeypatch, tmp_path, hashes)
    if problem == "partial": document["cases"].pop()
    elif problem == "duplicate": document["cases"][-1] = document["cases"][0]
    elif problem == "source_changed": (tmp_path / "source/training/tron1_tracking.py").write_text("changed source")
    elif problem == "bad_exit_capture":
        first = document["cases"][0]; path = tmp_path / first["receipt"]["path"]
        write(path, {"capture_complete": False, "owned_processes_and_contexts_absent": True, "source_bindings_unchanged": True})
        first["receipt"] = bind(path, tmp_path)
    elif problem == "artifact_hash": document["cases"][0]["files"]["trace"]["sha256"] = "0" * 64
    else: document["cases"][1]["log"] = document["cases"][0]["log"]
    write(plan, document)
    with pytest.raises(ValueError): assembly.assemble_nmpc(plan, tmp_path / "capture", tmp_path / "future")
    assert not (tmp_path / "capture").exists()
