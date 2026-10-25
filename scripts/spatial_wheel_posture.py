"""Analytic floor-tangent TRON1 postures with independent abduction and root RPY.

This is geometry and instantaneous material-point kinematics only. It does
not integrate physics, infer a normal load, or prove trajectory feasibility.
"""
from __future__ import annotations

import numpy as np

import generate_camber_reference as base
from independent_camber_reference import planar_angles


def root_quaternion(yaw,roll,pitch):
    """Hamilton quaternion for Rz(yaw) Ry(pitch) Rx(roll)."""
    cy,sy=np.cos(yaw/2),np.sin(yaw/2)
    cp,sp=np.cos(pitch/2),np.sin(pitch/2)
    cr,sr=np.cos(roll/2),np.sin(roll/2)
    return np.array([cy*cp*cr+sy*sp*sr,cy*cp*sr-sy*sp*cr,
                     cy*sp*cr+sy*cp*sr,sy*cp*cr-cy*sp*sr])


def axle_geometry(roll,pitch,abduction):
    """Yaw-normalized axle,world camber,and rolling-heading offset from body yaw."""
    angle=roll+np.asarray(abduction,dtype=float)
    axis=np.stack((np.sin(pitch)*np.sin(angle),np.cos(angle),np.cos(pitch)*np.sin(angle)),axis=-1)
    camber=np.arcsin(axis[...,2].clip(-1.,1.))
    heading=np.arctan2(-axis[...,0],axis[...,1])
    return axis,camber,heading


class Geometry(base.Generator):
    def __init__(self,model,source):
        super().__init__(model,source)
        self.pose(source['qpos_mujoco'][0])
        self.original_centers=self.data.geom_xpos[self.geoms].copy()-source['qpos_mujoco'][0,:3]
        self.links=[]
        self.opposite_leg_geoms=[]
        for side in ('L','R'):
            self.opposite_leg_geoms.append([i for i in range(model.ngeom)
                if model.geom_contype[i] and model.geom_conaffinity[i]
                and '_'+side+'_' in model.body(int(model.geom_bodyid[i])).name])
        for i,side in enumerate(('L','R')):
            o=model.body('abad_'+side+'_Link').pos.copy()
            h=model.body('hip_'+side+'_Link').pos.copy()
            thigh=model.body('knee_'+side+'_Link').pos.copy()
            shin=model.body('wheel_'+side+'_Link').pos.copy()
            hip,knee=model.joint('hip_'+side+'_Joint'),model.joint('knee_'+side+'_Joint')
            hs,ks=model.jnt_axis[hip.id,1],model.jnt_axis[knee.id,1]
            phi=np.arctan2(shin[2],shin[0])-np.arctan2(thigh[2],thigh[0])-ks*source['qpos_mujoco'][0,knee.qposadr.item()]
            self.links.append(dict(origin=o,hip=h,thigh=thigh,shin=shin,hip_joint=hip,knee_joint=knee,
                hip_sign=hs,knee_sign=ks,branch=int(np.sign(phi)),
                x=self.original_centers[i,0]-o[0]-h[0],
                lateral=h[1]+thigh[1]+shin[1]+model.geom_pos[self.geoms[i],1]))

    def posture(self,frame,roll,pitch,abduction,height):
        """Keep source root XY/yaw and wheel body-longitudinal coordinates.

        Both lowest ideal cylinder points are reconstructed onto z=0. Height
        is a posture variable,not an absolute-height policy gate or target.
        """
        values=np.r_[roll,pitch,abduction,height]
        if values.shape!=(5,) or not np.isfinite(values).all():raise ValueError('Require five finite posture values')
        q=self.source['qpos_mujoco'][frame].copy()
        yaw=2*np.arctan2(q[6],q[3])
        q[2]=height;q[3:7]=root_quaternion(yaw,roll,pitch);q[self.abads]=abduction
        ax,_,_=axle_geometry(roll,pitch,abduction)
        n=np.array([-np.sin(pitch),np.cos(pitch)*np.sin(roll),np.cos(pitch)*np.cos(roll)])
        for i,link in enumerate(self.links):
            radius,halfwidth=self.model.geom_size[self.geoms[i],:2]
            az=ax[i,2];rho=radius*np.sqrt(max(0.,1-az*az))+halfwidth*abs(az)
            angle=roll+abduction[i];denominator=np.cos(pitch)*np.cos(angle)
            if abs(denominator)<1e-8:raise ValueError('Degenerate leg reconstruction')
            fixed=n@link['origin']-np.sin(pitch)*(link['hip'][0]+link['x'])+np.cos(pitch)*np.sin(angle)*link['lateral']
            z=(rho-height-fixed)/denominator
            h,k=planar_angles(np.array([link['x'],z]),link['thigh'][[0,2]],link['shin'][[0,2]],link['branch'])
            q[link['hip_joint'].qposadr.item()]=h/link['hip_sign']
            q[link['knee_joint'].qposadr.item()]=k/link['knee_sign']
        for j in self.joint_ids:
            if self.model.jnt_limited[j] and not self.model.jnt_range[j,0]-1e-12<=q[self.model.jnt_qposadr[j]]<=self.model.jnt_range[j,1]+1e-12:
                raise ValueError('Authored joint limit')
        self.pose(q);points,axes,_=self.wheel_geometry()
        if np.max(abs(points[:,2]))>1e-9:raise ValueError('Lost geometric floor tangency')
        return q

    def nonwheel_clearance(self):
        lowest=[]
        for g in range(self.model.ngeom):
            if g in self.geoms or not self.model.geom_bodyid[g] or not self.model.geom_contype[g] or not self.model.geom_conaffinity[g]:continue
            mat,size=self.data.geom_xmat[g].reshape(3,3),self.model.geom_size[g]
            if self.model.geom_type[g]==self.mj.mjtGeom.mjGEOM_BOX:
                z=self.data.geom_xpos[g,2]-abs(mat[2])@size
            elif self.model.geom_type[g]==self.mj.mjtGeom.mjGEOM_CYLINDER:
                z=base.cylinder_lowest_point(self.data.geom_xpos[g],mat[:,2],*size[:2])[0][2]
            else:raise ValueError('Unsupported nonwheel collider')
            lowest.append(z)
        return min(lowest)

    def instantaneous_surface_motion(self,q,root_velocity_w,root_angular_velocity_w):
        """Frozen leg posture,source root twist,and longitudinal rolling spin.

        This is not the velocity of an interpolated posture sequence. Those
        leg/root shape derivatives must be recomputed for any real trajectory.
        """
        self.pose(q);points,axes,_=self.wheel_geometry()
        v=np.zeros(self.model.nv);v[:3]=root_velocity_w
        rotation=self.data.xmat[self.model.body('base_Link').id].reshape(3,3)
        v[3:6]=rotation.T@root_angular_velocity_w
        for i,(point,axis,body) in enumerate(zip(points,axes,self.wheels)):
            forward=np.cross(axis,[0.,0.,1.]);forward/=np.linalg.norm(forward)
            self.mj.mj_jac(self.model,self.data,self.jp,self.jr,point,body)
            coupling=forward@self.jp[:,self.spin_v[i]]
            if abs(coupling)<1e-5:raise ValueError('Degenerate rolling spin')
            v[self.spin_v[i]]=-(forward@self.jp@v)/coupling
        velocities=[]
        for point,body in zip(points,self.wheels):
            self.mj.mj_jac(self.model,self.data,self.jp,self.jr,point,body)
            velocities.append(self.jp@v)
        return v,np.asarray(velocities),points,axes

    def opposite_leg_clearance(self):
        """Explicit signed geom distance,independent of native contact masks."""
        return min(self.mj.mj_geomDistance(self.model,self.data,left,right,1.,None)
                   for left in self.opposite_leg_geoms[0] for right in self.opposite_leg_geoms[1])

    def frozen_shape_inverse_state(self,frame,roll,pitch,abduction,height):
        """Required generalized inertial/gravity force for constant shape.

        Root yaw/XY follow the original source while RPY offsets,height and
        leg shape are locally constant. Spin acceleration follows the same
        instantaneous longitudinal rolling calculation at neighboring times.
        No contact solver or physics integration is called.
        """
        def sample(index):
            q=self.posture(index,roll,pitch,abduction,height)
            source_v=self.source['qvel_mujoco'][index]
            v,slip,points,axes=self.instantaneous_surface_motion(q,source_v[:3],[0.,0.,source_v[5]])
            return q,v,slip,points,axes
        low,high=max(0,frame-1),min(len(self.time)-1,frame+1)
        before=sample(low);after=sample(high)
        q,v,slip,points,axes=sample(frame)
        acceleration=(after[1]-before[1])/(self.time[high]-self.time[low])
        self.data.qvel[:]=v;self.data.qacc[:]=acceleration
        self.mj.mj_comVel(self.model,self.data)
        required=np.empty(self.model.nv)
        self.mj.mj_rne(self.model,self.data,1,required)
        return dict(qpos=q,qvel=v,qacc=acceleration,required_generalized_force=required,
                    material_surface_velocity_w=slip,points_w=points,axes_w=axes)
