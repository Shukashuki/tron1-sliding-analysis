"""Observer provenance gates; CPU-only mocks do not validate actual PhysX."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import eval_physx_native_contact_diagnostic as observer
import physx_tire_runtime  # Load NumPy dependencies before scoped sys.modules mocks.


def test_production_pins_bind_completed_K_N_and_frozen_readback():
    for cohort, digest in observer.EVALUATOR_PINS.items():
        assert observer.sha256(ROOT/"scripts"/("eval_ppo_ground_contact_"+cohort+".py")) == digest
    assert observer.sha256(ROOT/"scripts/physx_tire_runtime.py") == observer.RUNTIME_PIN
    assert observer.sha256(ROOT/"scripts/nonlinear_tire_contact.py") == observer.ARITHMETIC_PIN


@pytest.mark.parametrize("corruption", ["observed_recording", "original_baseline", "parent_source"])
def test_corrupted_observation_or_parent_cannot_publish_contact_data(tmp_path, corruption):
    scripts = tmp_path/"base/scripts"; scripts.mkdir(parents=True)
    baseline = tmp_path/"baseline"; baseline.mkdir()
    output = tmp_path/"new"; output.mkdir()
    original = baseline/"physics_trajectory.npz"; original.write_bytes(b"original native recording")
    observed = output/"physics_trajectory.npz"
    observed.write_bytes(b"changed" if corruption == "observed_recording" else original.read_bytes())
    parent_file = scripts/"eval_ppo_ground_contact_v2k.py"; parent_file.write_text("# test-only parent\n")
    base = ModuleType("eval_ppo_ground_contact_v2k"); base.__file__ = str(parent_file)
    plan = {"args": SimpleNamespace(backend="physx", training_dir=tmp_path/"training", protocol=tmp_path/"protocol.json"),
            "mu": .7, "evaluator": tmp_path/"legacy.py", "legacy_root": tmp_path/"legacy", "output": output,
            "inherited_argv": []}
    plan["evaluator"].write_text("# fake legacy source\n")
    base.prepare_plan = lambda args: plan
    base.validate_output = lambda *args, **kwargs: {"recording_sha256": observer.sha256(original)}
    base.common = SimpleNamespace(verify_recording=lambda *args: None)
    physx = ModuleType("eval_ppo_backend_400hz_isaac")
    source = "def main():\n"
    if corruption == "original_baseline":
        source += "    from pathlib import Path\n    Path("+repr(str(original))+").write_bytes(b'changed while observing')\n"
    source += "    _new_finish_report({'status':'evaluated','deterministic_start_frame':0,'runtime_device_audit':{'status':'verified'}},{},_new_plan,{})\n"
    physx.materialize_source = lambda *args: source
    digest = observer.sha256(parent_file)
    argv = ["observer", "--base-snapshot", str(tmp_path/"base"), "--cohort", "v2k", "--baseline-dir", str(baseline)]
    with patch.dict(sys.modules, {base.__name__: base, physx.__name__: physx}), \
         patch.object(sys, "argv", argv), patch.dict(observer.EVALUATOR_PINS,
             {"v2k": "0"*64 if corruption == "parent_source" else digest}):
        old_path = list(sys.path)
        try:
            with pytest.raises(AssertionError, match=("parent evaluator" if corruption == "parent_source"
                    else "Original baseline changed" if corruption == "original_baseline" else "Observer changed")):
                observer.main()
        finally: sys.path[:] = old_path
    assert not (output/"native_contact_detail.npz").exists()
    assert not (output/"native_contact_diagnostic.json").exists()
