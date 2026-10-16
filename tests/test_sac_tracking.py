"""SAC numerical contracts, not a claim of a trained or simulator-verified policy."""
from pathlib import Path
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sac_tracking import SACAgent, SACConfig, SquashedActor, VectorReplay, soft_q_target, entropy_temperature_loss


def small_config(**kwargs):
    return SACConfig(hidden_sizes=(16, 16), **kwargs)


@pytest.mark.parametrize("activation,kind", [("elu", torch.nn.ELU), ("relu", torch.nn.ReLU)])
def test_explicit_actor_and_critic_activation_and_matching_default_widths(activation, kind):
    default = SACConfig()
    assert default.hidden_sizes == (256, 128, 128)
    assert default.activation == "elu"
    agent = SACAgent(small_config(activation=activation))
    assert [layer.out_features for layer in agent.actor.trunk if isinstance(layer, torch.nn.Linear)] == [16, 16]
    for network in (agent.actor.trunk, agent.critic.q1[0], agent.critic.q2[0]):
        assert isinstance(network[1], kind) and isinstance(network[3], kind)


@pytest.mark.parametrize("activation", ["tanh", "", None, True])
def test_invalid_activation_is_not_silently_replaced(activation):
    with pytest.raises(ValueError, match="activation"):
        small_config(activation=activation)


def transition(count, actor_dim=51, critic_dim=120, action_dim=8, start=0, device="cpu"):
    ids = torch.arange(start, start + count, dtype=torch.float32, device=device)
    return {"actor_obs": ids[:, None].expand(-1, actor_dim).clone(),
            "critic_obs": ids[:, None].expand(-1, critic_dim).clone(),
            "actions": torch.zeros(count, action_dim, device=device),
            "rewards": ids.clone(), "discounts": torch.full((count,), .99, device=device),
            "bootstrap_actor_obs": (ids + .5)[:, None].expand(-1, actor_dim).clone(),
            "bootstrap_critic_obs": (ids + .5)[:, None].expand(-1, critic_dim).clone(),
            "terminated": torch.zeros(count, dtype=torch.bool, device=device),
            "truncated": torch.zeros(count, dtype=torch.bool, device=device)}


def test_default_tron_dimensions_and_deterministic_bounded_export(tmp_path):
    torch.manual_seed(13)
    agent = SACAgent(small_config(), actor_mean=torch.linspace(-1, 1, 51), actor_scale=torch.full((51,), 2.))
    raw = torch.randn(7, 51)
    expected = agent.act(raw, deterministic=True)
    assert expected.shape == (7, 8) and (expected.abs() <= 1).all()
    assert agent.critic(torch.randn(7, 120), expected)[0].shape == (7,)
    torch.testing.assert_close(agent.act(raw, deterministic=True), expected)
    path = tmp_path / "actor_normalized.pt"
    agent.export_actor(path)
    loaded = torch.jit.load(str(path))
    torch.testing.assert_close(loaded(raw), expected)
    with torch.no_grad():
        h = agent.actor.trunk((raw - agent.actor.normalizer.mean) / agent.actor.normalizer.scale)
        torch.testing.assert_close(expected, torch.tanh(agent.actor.mu(h)))


def test_tanh_log_probability_matches_reference_and_stays_finite_at_saturation():
    actor = SquashedActor(small_config(actor_dim=3, critic_dim=4, action_dim=2))
    with torch.no_grad():
        for value in actor.parameters():
            value.zero_()
        actor.mu.bias.copy_(torch.tensor([.2, -.3]))
        actor.log_std.bias.fill_(-1.)
    raw = torch.zeros(11, 3)
    torch.manual_seed(7)
    action, logp = actor.sample(raw)
    pre = torch.atanh(action)
    distribution = torch.distributions.Normal(actor.mu.bias, actor.log_std.bias.exp())
    expected = (distribution.log_prob(pre) - torch.log1p(-action.square())).sum(-1)
    torch.testing.assert_close(logp, expected, atol=1e-5, rtol=1e-5)
    assert logp.shape == (11,) and torch.isfinite(logp).all()
    with torch.no_grad():
        actor.mu.bias.copy_(torch.tensor([100., -100.]))
    action, logp = actor.sample(raw)
    assert (action.abs() <= 1).all() and torch.isfinite(logp).all()
    logp.mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())


def test_targets_use_per_transition_discount_bootstrap_truncation_not_termination():
    reward = torch.tensor([1., 2., 3., 4.], requires_grad=True)
    discounts = torch.tensor([.9, .5, .7, .2])
    terminated = torch.tensor([False, False, True, True])
    truncated = torch.tensor([False, True, False, True])
    q = torch.tensor([10., 20., 30., 40.])
    logp = torch.tensor([-2., -3., -4., -5.])
    result = soft_q_target(reward, discounts, terminated, truncated, q, logp, .1)
    torch.testing.assert_close(result, torch.tensor([10.18, 12.15, 3., 4.]))
    assert not result.requires_grad


def test_temperature_loss_detaches_actor_log_probability():
    log_alpha = torch.tensor(-1., requires_grad=True)
    log_prob = torch.tensor([-2., -3.], requires_grad=True)
    entropy_temperature_loss(log_alpha, log_prob, -2.).backward()
    assert log_prob.grad is None
    torch.testing.assert_close(log_alpha.grad, torch.tensor(4.5))


def test_replay_wrap_large_vector_batches_and_checkpoint():
    replay = VectorReplay(5, actor_dim=3, critic_dim=4, action_dim=2)
    replay.add(**transition(3, 3, 4, 2))
    assert replay.size == 3 and replay.position == 3
    replay.add(**transition(4, 3, 4, 2, start=3))
    assert replay.size == 5 and replay.position == 2 and replay.total_added == 7
    torch.testing.assert_close(replay.storage["rewards"], torch.tensor([5., 6., 2., 3., 4.]))
    replay.add(**transition(8, 3, 4, 2, start=7))
    assert replay.position == 0 and replay.total_added == 15
    torch.testing.assert_close(replay.storage["rewards"], torch.arange(10., 15.))
    restored = VectorReplay(5, 3, 4, 2)
    restored.load_state_dict(replay.state_dict())
    assert restored.allocated_bytes == replay.allocated_bytes
    assert restored.position == replay.position
    for key in replay.storage:
        torch.testing.assert_close(restored.storage[key], replay.storage[key])
    sampled = restored.sample(30)
    assert set(sampled["rewards"].tolist()) <= set(range(10, 15))
    torch.testing.assert_close(sampled["bootstrap_actor_obs"][:, 0], sampled["rewards"] + .5)


def test_replay_requires_explicit_final_observations_and_boolean_flags():
    replay = VectorReplay(8)
    values = transition(2)
    values.pop("bootstrap_actor_obs")
    with pytest.raises(TypeError, match="bootstrap_actor_obs"):
        replay.add(**values)
    values = transition(2)
    values["truncated"] = torch.tensor([0., 1.])
    with pytest.raises(ValueError, match="boolean"):
        replay.add(**values)
    values = transition(2)
    values["discounts"][0] = 1.01
    with pytest.raises(ValueError, match="discount"):
        replay.add(**values)
    assert replay.size == 0


def test_truncated_replay_preserves_caller_supplied_final_observations():
    replay = VectorReplay(4, 3, 4, 2)
    values = transition(2, 3, 4, 2)
    values["truncated"][1] = True
    values["bootstrap_actor_obs"][1].fill_(123.)
    values["bootstrap_critic_obs"][1].fill_(456.)
    replay.add(**values)
    assert replay.storage["bootstrap_actor_obs"][1, 0] == 123.
    assert replay.storage["bootstrap_critic_obs"][1, 0] == 456.
    assert not replay.storage["terminated"][1]


def test_update_finite_and_polyak_exact_with_privileged_critics():
    torch.manual_seed(5)
    config = small_config(tau=.1)
    agent = SACAgent(config)
    replay = VectorReplay(32)
    replay.add(**transition(16))
    old_target = [p.detach().clone() for p in agent.target_critic.parameters()]
    old_actor = [p.detach().clone() for p in agent.actor.parameters()]
    old_alpha = agent.log_alpha.detach().clone()
    metrics = agent.update(replay.sample(8))
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["updates"] == 1 and metrics["alpha"] > 0
    assert not torch.equal(old_alpha, agent.log_alpha)
    assert any(not torch.equal(before, after) for before, after in zip(old_actor, agent.actor.parameters()))
    for before, target, online in zip(old_target, agent.target_critic.parameters(), agent.critic.parameters()):
        torch.testing.assert_close(target, before * .9 + online * .1)
        assert not target.requires_grad
    assert all(p.grad is None for p in agent.critic.parameters())


def test_checkpoint_resumes_optimizers_temperature_replay_and_rng(tmp_path):
    torch.manual_seed(41)
    config = small_config(actor_dim=3, critic_dim=4, action_dim=2)
    agent = SACAgent(config, actor_mean=[1., 2., 3.], actor_scale=[2., 3., 4.])
    replay = VectorReplay(16, 3, 4, 2)
    replay.add(**transition(9, 3, 4, 2))
    agent.update(replay.sample(6))
    path = tmp_path / "checkpoint.pt"
    agent.save(path, replay)
    expected = agent.update(replay.sample(6))
    expected_actor = [p.detach().clone() for p in agent.actor.parameters()]
    restored = SACAgent(config)
    restored_replay = VectorReplay(16, 3, 4, 2)
    restored.load(path, restored_replay)
    actual = restored.update(restored_replay.sample(6))
    assert actual == pytest.approx(expected, abs=1e-7)
    for before, after in zip(expected_actor, restored.actor.parameters()):
        torch.testing.assert_close(before, after)
    torch.testing.assert_close(agent.log_alpha, restored.log_alpha)
    assert restored.updates == 2
    assert restored.alpha_optimizer.state_dict()["state"]


@pytest.mark.skipif(os.environ.get("TRON1_SAC_TEST_CUDA") != "1" or not torch.cuda.is_available(),
                    reason="CUDA test requires explicit TRON1_SAC_TEST_CUDA=1 and an available GPU")
def test_replay_and_update_on_cuda():
    replay = VectorReplay(16, 3, 4, 2, device="cuda")
    replay.add(**transition(7, 3, 4, 2, device="cuda"))
    replay.add(**transition(19, 3, 4, 2, start=7, device="cuda"))
    assert replay.size == 16 and replay.position == 10
    assert set(replay.storage["rewards"].cpu().tolist()) == set(range(10, 26))
    agent = SACAgent(small_config(actor_dim=3, critic_dim=4, action_dim=2), device="cuda")
    assert all(math.isfinite(value) for value in agent.update(replay.sample(4)).values())
