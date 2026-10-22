"""No-GL invariants for the damping recorded-state renderer."""
from pathlib import Path
import ast
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import render_ppo_contact_damping as render


def test_reference_camera_is_common_and_does_not_read_actual_poses():
    reference = render.load_reference(render.DEFAULT_REFERENCE)
    target = render.common_camera_target(reference, 2.4)
    assert target == pytest.approx(reference["position"][120]+[0., 0., -.24])
    assert render.common_camera_target(reference, 6.98) == pytest.approx(reference["position"][-1]+[0., 0., -.24])
    assert np.array_equal(np.arange(0, 1397, 4)*.005, np.arange(350)*.02)


def test_no_new_dynamics_or_inference_calls():
    source = Path(render.__file__).read_text(); tree = ast.parse(source)
    calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else "" for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not set(calls).intersection({"mj_step", "mj_forward", "step", "infer", "predict", "learn"})
    assert "_pose" in calls and "validate_fk" in calls


def test_fresh_output_required_before_inputs_or_gl(tmp_path):
    args = render.parser().parse_args(["--input-root", "missing", "--output-dir", str(tmp_path), "--skip-video"])
    with pytest.raises(ValueError, match="Preserve existing"):
        render.run(args)
