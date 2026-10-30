"""Restore a completed PPO screen without resetting its optimizer or noise.

This continues learning in a fresh environment. Simulator state, RNG state and
adaptive phase-sampling history were not checkpointed and are not restored.
Iteration and transition counters describe the new segment, starting at zero.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path


def checked_bytes(path, expected):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Continuation input hash changed: ' + str(path))
    return payload


def state_digest(value):
    """Device-independent digest of tensor values and optimizer metadata."""
    import torch
    result = hashlib.sha256()

    def visit(item):
        result.update(type(item).__name__.encode() + b':')
        if isinstance(item, torch.Tensor):
            result.update(str((item.dtype, tuple(item.shape))).encode())
            result.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            result.update(repr(item).encode())
        else:
            raise TypeError('Unsupported checkpoint state: ' + type(item).__name__)
        result.update(b';')

    visit(value)
    return result.hexdigest()


def matching_recipe(source, target):
    """Permit only the checkpoint source, new iteration budget and metadata."""
    values = []
    for original in (source, target):
        cfg = copy.deepcopy(original)
        for key in ('warm_start', 'initialization', 'training_continuation'):
            cfg.pop(key, None)
        iterations = cfg['formal'].pop('iterations')
        if type(iterations) is not int or iterations <= 0:
            raise ValueError('Require a positive integer training budget')
        values.append(cfg)
    if values[0] != values[1]:
        raise ValueError('Continuation must preserve rewards, physics, DR and PPO settings')
    if source['ppo']['schedule'] != 'fixed':
        raise ValueError('Only the audited fixed learning-rate schedule is supported')


def initialize_continuation(runner, protocol):
    import torch
    settings = protocol['training_continuation']
    expected_keys = {'kind', 'source_report', 'source_report_sha256',
                     'source_protocol', 'source_protocol_sha256'}
    if (set(settings) != expected_keys
            or settings['kind'] != 'hockey_optimizer_continuation_v1'):
        raise ValueError('Unknown continuation settings')
    source = json.loads(checked_bytes(settings['source_protocol'], settings['source_protocol_sha256']))
    report = json.loads(checked_bytes(settings['source_report'], settings['source_report_sha256']))
    matching_recipe(source, protocol)
    parent = protocol['warm_start']
    prior_iterations = source['formal']['iterations']
    prior_transitions = prior_iterations * source['formal']['num_envs'] * source['ppo']['steps_per_env']
    if (report['status'] != 'training_completed_not_task_success'
            or report['config_sha256'] != settings['source_protocol_sha256']
            or report['transitions'] != prior_transitions
            or report['worlds'] != source['formal']['num_envs']
            or report['bilateral_audit']['fault_seen']
            or report['checkpoint_sha256'] != parent['checkpoint_sha256']
            or report['actor_sha256'] != parent['actor_sha256']):
        raise ValueError('Continuation requires an intact, fully completed source screen')
    if runner.tot_timesteps != 0 or runner.current_learning_iteration != 0:
        raise ValueError('New segment counters must start at zero')
    if getattr(runner.alg, 'rnd', None):
        raise ValueError('RND continuation was not audited')
    checkpoint = torch.load(io.BytesIO(checked_bytes(parent['checkpoint'], parent['checkpoint_sha256'])),
                            map_location='cpu', weights_only=True)
    if (set(checkpoint) != {'model_state_dict', 'optimizer_state_dict', 'iter', 'infos'}
            or checkpoint['iter'] != prior_iterations - 1):
        raise ValueError('Checkpoint iteration or saved-state schema differs')
    model_state = checkpoint['model_state_dict']
    optimizer_state = checkpoint['optimizer_state_dict']
    if 'log_std' not in model_state or not optimizer_state['state']:
        raise ValueError('Missing learned exploration or optimizer state')
    for value in model_state.values():
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite learned model state')
    for state in optimizer_state['state'].values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError('Nonfinite optimizer state')
    learning_rate = source['ppo']['learning_rate']
    if (runner.alg.learning_rate != learning_rate
            or any(group['lr'] != learning_rate for group in optimizer_state['param_groups'])):
        raise ValueError('Optimizer learning rate differs from the fixed recipe')
    model_hash, optimizer_hash = state_digest(model_state), state_digest(optimizer_state)
    runner.alg.policy.load_state_dict(model_state, strict=True)
    runner.alg.optimizer.load_state_dict(optimizer_state)
    if (state_digest(runner.alg.policy.state_dict()) != model_hash
            or state_digest(runner.alg.optimizer.state_dict()) != optimizer_hash):
        raise ValueError('Checkpoint restore changed model or optimizer values')
    if runner.tot_timesteps != 0 or runner.current_learning_iteration != 0:
        raise ValueError('Checkpoint restore changed segment counters')
    return dict(checkpoint_sha256=parent['checkpoint_sha256'], parent_iteration=checkpoint['iter'],
        source_report_sha256=settings['source_report_sha256'],
        source_protocol_sha256=settings['source_protocol_sha256'],
        actor_critic_and_normalizers_loaded=True, optimizer_loaded=True,
        exploration_reset=False, exploration_std=runner.alg.policy.log_std.exp().detach().cpu().tolist(),
        initial_model_state_sha256=model_hash, initial_optimizer_state_sha256=optimizer_hash,
        fresh_transition_budget=True, local_iteration_origin=0,
        prior_same_recipe_transitions=report.get('same_recipe_total_transitions', prior_transitions),
        source_segment_transitions=prior_transitions,
        simulator_state_restored=False, rng_state_restored=False,
        adaptive_phase_sampling_history_restored=False, bitwise_training_resume_claimed=False)
