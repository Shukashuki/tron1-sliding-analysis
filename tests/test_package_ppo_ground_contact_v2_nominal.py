"""Fresh publication, full-clock identities, no weight copies or claim inflation."""
from pathlib import Path
import sys
from types import SimpleNamespace
import copy
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import package_ppo_ground_contact_v2_nominal as mod


def test_existing_output_rejected_before_inputs(tmp_path):
    with pytest.raises(ValueError, match="fresh output"):
        mod.run(SimpleNamespace(output_dir=tmp_path, input_root=tmp_path/"missing"))


def test_allowlist_only_json_training_proofs():
    assert all(name.endswith(".json") for name in mod.PROOF_FILES)
    assert {"policy.pt", "model_final.pt"}.isdisjoint(mod.PROOF_FILES)
    assert set(mod.BOUND_PROOFS) <= set(mod.PROOF_FILES)


def test_source_snapshot_rejects_binary_or_indirect_member(tmp_path):
    archive = tmp_path/"bad.tar.gz"
    binary = tmp_path/"actor.pt"; binary.write_bytes(b"no")
    with tarfile.open(archive, "w:gz") as stream: stream.add(binary, arcname="actor.pt")
    manifest = tmp_path/"manifest.json"
    mod.write_json(manifest, {"archive_sha256": mod.sha256(archive), "no_models_weights_binaries_credentials": True,
        "files": [{"path": "actor.pt", "size_bytes": 2, "sha256": mod.sha256(binary)}]})
    with pytest.raises(ValueError, match="non-source"):
        mod.verify_source_snapshot(archive, manifest)


def test_local_source_archive_dependencies_are_pure_and_include_renderer():
    ledger = mod.local_source_ledger()
    assert "scripts/render_ppo_ground_contact_v2.py" in ledger
    assert "scripts/analyze_ppo_ground_contact_v2.py" in ledger
    assert all(Path(name).suffix in (".py", ".json") for name in ledger)


def test_actual_recorded_compact_metrics_and_wrong_identity_rejection():
    folder = mod.DEFAULT_ROOT/"ppo_dual_engine_analysis_v1"
    if not (folder/"summary.json").exists(): pytest.skip("Formal local records not present")
    summary = mod.read_json(folder/"summary.json")
    actual = mod.recompute_compact(folder/"native_states.npz", summary)
    assert set(actual) == set(mod.CASE_ORDER)
    bad = copy.deepcopy(summary)
    bad["rows"][0]["new_V2"]["windows"]["core"]["tracking"]["xy_rmse_m"] += .001
    with pytest.raises(ValueError, match="recomputation"):
        mod.recompute_compact(folder/"native_states.npz", bad)


def test_actual_render_bindings_and_reject_wrong_labels_clock_and_camera():
    root = mod.DEFAULT_ROOT
    if not (root/"ppo_render_v1/provenance.json").exists(): pytest.skip("Render not completed")
    render = mod.read_json(root/"ppo_render_v1/provenance.json")
    summary = mod.read_json(root/"ppo_dual_engine_analysis_v1/summary.json")
    mod.verify_render(render, summary, root/"ppo_render_v1")
    for key, wrong in (("case_order", list(reversed(mod.CASE_ORDER))), ("video_frames", 349),
                       ("camera_zoom", True), ("goal_completion_claimed", True)):
        bad = copy.deepcopy(render); bad[key] = wrong
        with pytest.raises(ValueError): mod.verify_render(bad, summary, root/"ppo_render_v1")


def test_scope_does_not_conflate_completed_training_and_completed_goal():
    text = Path(mod.__file__).read_text()
    assert '"goal_complete": False' in text
    assert '"formal_training_and_validation_complete": True' in text
    assert "FORMAL_PPO_COMPLETED_NOMINAL_DUAL_ENGINE_CONTACT_FAILURE" in text
    assert "legacy evaluator reward logging is NOT numerically compared with CEM returns" in text
