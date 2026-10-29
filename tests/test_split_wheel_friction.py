from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import physx_mjlab_recipe as recipe
from wheel_friction_capacity import capacity_ratios
from test_physx_mjlab_recipe import fake_env, fake_cfg

@pytest.mark.parametrize('ratio',[.9,0,True,float('nan'),float('inf')])
def test_invalid_static_dynamic_ratio_rejected(ratio):
    with pytest.raises(ValueError):recipe.recipe_manifest(static_dynamic_ratio=ratio)


def test_fixed_pair_changes_wheels_and_rejects_wrong_static_readback():
    env=fake_env(4)
    recipe.initialize_contact_materials(env,None,training=False,fixed_mu=.7,static_dynamic_ratio=9/7)
    recipe.initialize_encoder_bias(env,None,training=False)
    a=recipe.audit_physx_mjlab_recipe(env,check_usd=False)
    assert a['wheel_mu_static_range']==pytest.approx([.9,.9])
    assert a['wheel_mu_dynamic_range']==pytest.approx([.7,.7])
    assert not a['recipe']['wheel_contact_friction']['physx_static_equals_dynamic']
    values=env.scene['robot'].root_physx_view.materials
    torch.testing.assert_close(values[:,:7,:2],torch.full((4,7,2),.7))
    values[0,7,0]=.7
    with pytest.raises(RuntimeError,match='readback'):recipe.audit_physx_mjlab_recipe(env,check_usd=False)


def test_training_draw_keeps_dynamic_DR_and_all_18_slice_materials_share_pair():
    old=fake_env(32);torch.manual_seed(42);recipe.initialize_contact_materials(old,None)
    env=fake_env(32);robot=env.scene['robot'];view=robot.root_physx_view
    view.max_shapes=25;view.materials=torch.zeros((32,25,3))
    from types import SimpleNamespace as NS
    robot._physics_sim_view.create_rigid_body_view=lambda path:NS(max_shapes=9 if 'wheel_' in path else 0 if 'imu_' in path else 1)
    torch.manual_seed(42);recipe.initialize_contact_materials(env,None,static_dynamic_ratio=9/7)
    recipe.initialize_encoder_bias(env,None)
    audit=recipe.audit_physx_mjlab_recipe(env,check_usd=False)
    assert len(audit['wheel_shape_slots'])==18
    torch.testing.assert_close(env.mjlab_recipe_mu,old.mjlab_recipe_mu,rtol=0,atol=0)
    expected=torch.stack((env.mjlab_recipe_mu*9/7,env.mjlab_recipe_mu),-1)[:,None].expand(-1,18,-1)
    torch.testing.assert_close(view.materials[:,7:,:2],expected)
    torch.testing.assert_close(view.materials[:,:7,:2],torch.ones((32,7,2)))


def test_installer_passes_split_pair_into_startup_event(monkeypatch):
    from types import ModuleType,SimpleNamespace
    module=ModuleType('isaaclab.managers');module.EventTermCfg=SimpleNamespace
    monkeypatch.setitem(sys.modules,'isaaclab.managers',module)
    cfg=fake_cfg();recipe.install_physx_mjlab_recipe(cfg,training=False,fixed_mu=.7,static_dynamic_ratio=9/7)
    assert cfg.events.mjlab_contact_material.params['static_dynamic_ratio']==9/7
    assert cfg.scene.terrain.physics_material.static_friction==cfg.scene.terrain.physics_material.dynamic_friction==1


def test_two_capacity_ratios_require_touch_and_do_not_identify_solver_state():
    result=capacity_ratios(np.array([100.,100.,100.,0.]),np.array([[0,70,0],[0,80,0],[0,70,0],[0,0,0]]),[1,1,0,1],.9,.7)
    assert result['rho_dynamic'][0]==pytest.approx(1)
    assert result['rho_static'][0]==pytest.approx(7/9)
    assert result['rho_dynamic'][1]>1 and result['rho_static'][1]<1
    assert np.isnan(result['rho_static'][2:]).all()
