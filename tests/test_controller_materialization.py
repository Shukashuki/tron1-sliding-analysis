from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import materialize_controller_source as materializer


def test_controller_source_preserves_pins_and_adds_opt_in_scripted_sac(tmp_path):
    source = tmp_path / "base"
    names = ("training/tron1_drift.py", "scripts/eval_tracking.py", "scripts/eval_tracking_mujoco.py")
    for name in names:
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/"upstream/tron1-learning-from-human"/name).read_bytes())
    hashes = {name: materializer.full.wide.sha(source/name) for name in names}
    output = tmp_path/"controller_source"
    audit = materializer.materialize(source, output)
    assert audit["full_horizon_diagnostic"] is True
    assert audit["controller_algorithms"] == ["PPO", "SAC"]
    assert hashes == {name: materializer.full.wide.sha(source/name) for name in names}
    for name, expected in audit["changed_output_sha256"].items():
        assert materializer.full.wide.sha(output/name) == expected
        compile((output/name).read_text(), name, "exec")
    code = (output/"scripts/eval_tracking_friction_isaac.py").read_text()
    assert "if runner_cfg.get(\"algorithm\") == \"SAC\":" in code
    assert "runner = OnPolicyRunner" in code
    assert 'contract["scripted_actor_provenance"] = runner.inference_provenance' in code
    assert 'if args.allow_smoke_policy:' in code
    assert 'if metadata.get("height_regularization") is not None:' in code
    with pytest.raises(FileExistsError):
        materializer.materialize(source, output)
