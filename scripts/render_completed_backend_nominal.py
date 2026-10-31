"""Four horizontal top views of the hash-verified matched PPO nominal cells."""
import argparse
import json
from pathlib import Path
import numpy as np
import analyze_ppo_backend_400hz_matrix as shared
from replay_physx_actions_mujoco import load_evaluator


def run(a):
    summary=json.loads(a.summary.read_text())
    reference=shared.reference_data(a.reference)
    assert shared.sha256(a.model)==shared.INPUT_PINS['model']
    loader=load_evaluator(a.legacy_evaluator)
    model,_=loader.load_model(a.model,no_visual_mesh=True)
    fk=shared.WheelKinematics(model)
    cases=[]
    for origin in ('physx','mjlab'):
        for engine in ('physx','mujoco'):
            folder=a.trace_root/(origin+'_trained'+('_mujoco' if engine=='mujoco' else ''))
            row=next(r for r in summary['rows'] if r['training_backend']==origin and r['engine_tag']==engine and r['mu']==.7)
            recording=folder/('physics_trajectory.npz' if engine=='physx' else 'rollout.npz')
            assert shared.sha256(recording)==row['input_hashes']['recording']
            assert shared.sha256(folder/'report.json')==row['input_hashes']['report']
            with np.load(recording,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            trace,_=shared.analyze_case(raw,engine,reference,fk)
            cases.append((origin,engine,trace,row))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,4,figsize=(20,5),sharex=True,sharey=True)
    xy=np.concatenate([reference['position'][:,:2],*[t['position'][:,:2] for _,_,t,_ in cases]])
    low,high=xy.min(axis=0)-.35,xy.max(axis=0)+.35
    for ax,(origin,engine,trace,row) in zip(axs,cases):
        p=trace['position'];times=trace['time_s']
        indexes=np.array([np.argmin(abs(times-t)) for t in (2.,3.,4.)])
        ax.plot(reference['position'][:,0],reference['position'][:,1],'--',color='gray',label='Reference')
        ax.plot(p[:,0],p[:,1],color='#15956c',label='Base path')
        body=shared.quaternion_rotate(trace['quaternion'],np.array([1.,0.,0.]))
        axle=shared.quaternion_rotate(trace['wheel_quaternion_wxyz'],np.array([0.,1.,0.]))
        rolling=np.cross(axle,np.array([0.,0.,1.]))
        for positions,vectors,color,label in [(p,body,'#1464b4','Body heading'),
                (p,trace['com_velocity'],'#ed9518','Velocity direction'),
                (trace['wheel_pos'][:,0],rolling[:,0],'#d62728','Left wheel plane'),
                (trace['wheel_pos'][:,1],rolling[:,1],'#9467bd','Right wheel plane')]:
            norm=np.linalg.norm(vectors[:,:2],axis=1);idx=indexes[norm[indexes]>1e-6]
            v=.18*vectors[idx,:2]/norm[idx,None]
            ax.quiver(positions[idx,0],positions[idx,1],v[:,0],v[:,1],color=color,label=label,
                angles='xy',scale_units='xy',scale=1,width=.006)
            if 'wheel' in label:
                ax.quiver(positions[idx,0],positions[idx,1],-v[:,0],-v[:,1],color=color,
                    angles='xy',scale_units='xy',scale=1,width=.006)
        name='PhysX' if origin=='physx' else 'MJWarp'
        destination='PhysX' if engine=='physx' else 'CPU MuJoCo'
        error=row['windows']['core']['tracking']['xy_rmse_m']
        ax.set(title=f'{name}-trained → {destination}\nCore XY RMSE {error:.3f} m',
            xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='World X (m)',ylabel='World Y (m)')
        ax.set_aspect('equal');ax.grid(alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=6,fontsize=9)
    fig.suptitle('400 Hz PPO | mu=0.7 | same world limits, no alignment | arrows at 2/3/4 s, fixed length\nN=1 per cell; path tracking is not continuous-contact certification')
    fig.tight_layout(rect=(0,.16,1,.89));fig.savefig(a.output,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('summary','trace-root','reference','model','legacy-evaluator','output'):
        p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
