"""Same MPPI search and cost with predicted first-order command smoothing.

The filter is upstream of the unchanged500/10 PD and DC torque envelope.
Prediction and the continuous plant use the same causal filter. This changes
the planning command decoder, not recorded poses or physical motor parameters.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import generate_mppi_contact_reference as base

PARENT_SHA='8843963f280afcc7cca6b354ad7a39cd4a7c6b05ba7d754e93d812adc0a66c65'


def filtered_controls(commands,previous,alpha):
    commands=np.asarray(commands,dtype=float);previous=np.asarray(previous,dtype=float);alpha=np.asarray(alpha,dtype=float)
    if commands.ndim!=3 or commands.shape[-1]!=8 or previous.shape!=(8,) or alpha.shape!=(8,) or np.any(alpha<=0) or np.any(alpha>1):
        raise ValueError('Invalid causal filter shape/coefficients')
    out=np.empty_like(commands);last=np.broadcast_to(previous,(len(commands),8)).copy()
    for step in range(commands.shape[1]):
        last=alpha*np.clip(commands[:,step],-1,1)+(1-alpha)*last;out[:,step]=last
    return out


def materialize(source):
    if hashlib.sha256(source.encode()).hexdigest()!=PARENT_SHA:raise ValueError('Parent generator changed')
    replacements=[
        ("cfg['kind'] != 'native_mppi_contact_reference_v1'", "cfg['kind'] != 'native_mppi_smooth_contact_reference_v1'"),
        ("        n = len(actions)\n", "        n = len(actions)\n        actions = filtered_controls(actions, previous_action, self.cfg['action_filter_alpha'])\n"),
        ("                decisions.append(dict(frame=frame,action=action.tolist(),**info))\n",
         "                raw_command=action.copy()\n                action=filtered_controls(raw_command[None,None],previous,cfg['action_filter_alpha'])[0,0]\n                info['raw_command_before_filter']=raw_command.tolist()\n                info['command_decoder']='clipped MPPI command then causal first-order filter then unchanged PD/DC'\n                decisions.append(dict(frame=frame,action=action.tolist(),**info))\n"),
        ("        source_sha256=sha(__file__), mppi_source_sha256=sha(ROOT/'scripts/mppi_shooting.py'),\n",
         "        source_sha256=sha(__file__), base_generator_source_sha256='"+PARENT_SHA+"', command_filter_alpha=cfg['action_filter_alpha'], mppi_source_sha256=sha(ROOT/'scripts/mppi_shooting.py'),\n"),
    ]
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('Materialization anchor is not unique: '+old)
        source=source.replace(old,new)
    return source


def main():
    source=materialize(Path(base.__file__).read_text())
    namespace=dict(__name__='_smooth_mppi_reference',__file__=str(Path(__file__).resolve()),filtered_controls=filtered_controls)
    exec(compile(source,'<smooth_mppi_reference>','exec'),namespace)
    args=namespace['parser']().parse_args()
    namespace['run'](args)


if __name__=='__main__':main()
