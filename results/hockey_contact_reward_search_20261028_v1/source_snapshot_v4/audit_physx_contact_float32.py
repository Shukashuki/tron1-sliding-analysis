"""Replay archived native-buffer readbacks with the recorder's float32 arithmetic.

This audits instrumentation, not physical success. Normalized float64 geometry
remains the independent physical calculation. Near a vertical axle, quaternion
rounding and normalization can amplify its difference from the recorded readback.
"""
import numpy as np


def replay_contact_readback(trace, raw, mu):
    import torch
    from physx_reference_math import reduce_contacts
    n = len(trace['root_pos'])
    indices = raw['physical_sample_indices']
    if not np.array_equal(indices, np.arange(1, n)):
        raise ValueError('Missing native physics samples')
    mapping = dict(normal_fz='normal_fz', normal_point_lateral_rms='lateral_slip_rms',
                   anchor_power='friction_anchor_power', friction_force='friction_force_w',
                   aggregate_utilization='utilization_proxy')
    result = {key: np.zeros_like(trace[value]) for key, value in mapping.items()}
    names = trace['body_names'].astype(str).tolist()
    with torch.inference_mode():
        for side, name in enumerate(('wheel_L_Link', 'wheel_R_Link')):
            bid = names.index(name)
            def pack(kind, fields, count_index, start_index):
                values = [torch.from_numpy(raw[f'w{side}_{kind}_buffer_{k}'].copy()) for k in range(fields)]
                capacity = values[0].shape[1]
                for k in range(count_index):
                    values[k] = values[k].reshape(-1, *values[k].shape[2:])
                counts = values[count_index].reshape(n-1, 1)
                starts = values[start_index].reshape(n-1, 1)
                if bool(((counts < 0) | (counts > 128) | (starts < 0) | (starts+counts > capacity)).any()):
                    raise ValueError('Native buffer row range invalid')
                values[count_index] = counts
                values[start_index] = starts + torch.arange(n-1)[:, None]*capacity
                return tuple(values)
            normal = pack('normal', 6, 4, 5)
            friction = pack('friction', 4, 2, 3)
            states = []
            for key in ('body_com_pos', 'body_com_vel', 'body_omega', 'body_quat'):
                value = trace[key][indices, bid]
                if value.dtype != np.float32:
                    raise ValueError('Recorder replay requires archived float32 states')
                if key == 'body_com_pos':
                    value = value + trace['environment_origin_w']
                states.append(torch.from_numpy(value.copy()))
            reduced = reduce_contacts(normal, friction, *states, torch.full((n-1,), mu, dtype=torch.float32))
            if bool(reduced['fault'].any()):
                raise ValueError('Invalid native replay sample')
            for key, recorded in mapping.items():
                result[key][indices, side] = reduced[recorded].numpy()
    return result
