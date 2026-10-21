// Persistent CPU PPO plant, not the cold-reset CEM prediction implementation.
// Only Python-owned, live MuJoCo 3.8.1 objects enter this private pointer bridge.
// There are no callbacks or error-handler replacements: a native fatal error
// terminates the isolated training process, never an embedding controller.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

static_assert(mjVERSION_HEADER == 3008001, "Requires MuJoCo 3.8.1");
static_assert(sizeof(mjtNum) == sizeof(double), "Requires double precision");

namespace {
constexpr int kNq=15, kNv=14, kNu=8, kBodies=9, kSubsteps=4;
constexpr int kSnapshot=146, kOutput=287, kIds=33;
int fail(char* error, size_t length, const char* message) {
  if (error && length) std::snprintf(error, length, "%s", message);
  return 1;
}
bool finite(const double* value, size_t count) {
  for (size_t i=0; i<count; ++i) if (!std::isfinite(value[i])) return false;
  return true;
}
bool no_callbacks() {
  return !(mjcb_control || mjcb_passive || mjcb_sensor || mjcb_contactfilter ||
           mjcb_act_dyn || mjcb_act_gain || mjcb_act_bias);
}
bool warning(const mjData* d) {
  for (int i=0; i<mjNWARNING; ++i) if (d->warning[i].number) return true;
  return false;
}
bool external_force(const mjModel* m, const mjData* d) {
  for (int i=0; i<kNv; ++i) if (d->qfrc_applied[i] != 0) return true;
  for (int i=0; i<6*m->nbody; ++i) if (d->xfrc_applied[i] != 0) return true;
  return false;
}
void torque(const mjData* d, const int* ids, const double* q0,
            const double* action, double* out) {
  for (int j=0; j<kNu; ++j) {
    const double a=std::clamp(action[j], -1., 1.);
    const double dq=d->qvel[ids[kNu+j]];
    const double effort=j<6 ? 80. : 12., speed=j<6 ? 15. : 100.;
    const double requested=j<6 ? 500.*(q0[j]+a-d->qpos[ids[j]])-10.*dq : 12.*a;
    const double v=std::clamp(dq, -2.*speed, 2.*speed);
    const double lower=std::max(effort*(-1.-v/speed), -effort);
    const double upper=std::min(effort*(1.-v/speed), effort);
    out[j]=std::clamp(requested, lower, upper);
  }
}
void snapshot(const mjModel* m, const mjData* d, const int* ids, double* out) {
  std::memcpy(out, d->qpos, kNq*sizeof(double));
  std::memcpy(out+kNq, d->qvel, kNv*sizeof(double));
  double* pos=out+kNq+kNv;
  double* quat=pos+3*kBodies;
  double* linear=quat+4*kBodies;
  double* angular=linear+3*kBodies;
  for (int i=0; i<kBodies; ++i) {
    const int body=ids[3*kNu+i];
    std::memcpy(pos+3*i, d->xpos+3*body, 3*sizeof(double));
    std::memcpy(quat+4*i, d->xquat+4*body, 4*sizeof(double));
    // BODY objectVelocity is original unmerged body-COM velocity in world axes.
    double velocity[6];
    mj_objectVelocity(m, d, mjOBJ_BODY, body, velocity, 0);
    std::memcpy(linear+3*i, velocity+3, 3*sizeof(double));
    std::memcpy(angular+3*i, velocity, 3*sizeof(double));
  }
  out[kOutput-1]=d->time;
}
void contacts(const mjModel* m, const mjData* d, const int* ids, double* out) {
  std::fill(out, out+kBodies*3, 0.);
  for (int c=0; c<d->ncon; ++c) {
    double wrench[6], world[3];
    mj_contactForce(m, d, c, wrench);
    const double* frame=d->contact[c].frame;
    for (int axis=0; axis<3; ++axis)
      world[axis]=frame[axis]*wrench[0]+frame[3+axis]*wrench[1]+frame[6+axis]*wrench[2];
    for (int side=0; side<2; ++side) {
      const int geom=d->contact[c].geom[side];
      if (geom<0) continue;
      const int body=m->geom_bodyid[geom];
      for (int i=0; i<kBodies; ++i) if (ids[3*kNu+i]==body)
        for (int axis=0; axis<3; ++axis)
          out[3*i+axis]+=(side ? 1. : -1.)*world[axis];
    }
  }
}
}  // namespace

extern "C" {
int tron1_ppo_abi() noexcept { return 1; }
int tron1_ppo_header_version() noexcept { return mjVERSION_HEADER; }
int tron1_ppo_runtime_version() noexcept { return mj_version(); }
int tron1_ppo_real_size() noexcept { return sizeof(mjtNum); }
int tron1_ppo_output_width() noexcept { return kOutput; }
int tron1_ppo_warning_width() noexcept { return mjNWARNING; }
int tron1_ppo_openmp() noexcept {
#ifdef _OPENMP
  return 1;
#else
  return 0;
#endif
}
const char* tron1_ppo_linked_mujoco_path() noexcept {
  Dl_info info;
  return dladdr(reinterpret_cast<void*>(mj_version), &info) ? info.dli_fname : nullptr;
}

// errors: persistent per-env status, zeroed only by Python's explicit reset.
// 1=preexisting warning, 2=new warning, 3=external force, 4=nonfinite state,
// 5=clock discontinuity (including a MuJoCo automatic numerical reset).
int tron1_ppo_batch(const mjModel* const* models, mjData* const* datas,
                    int batch, int threads, int stepping, const int* indices,
                    const double* q0s, const double* actions, double* output,
                    size_t output_length, int* errors, int* completed,
                    int* warning_counts, int* warning_lastinfo,
                    char* error, size_t error_length) noexcept {
  try {
    if (!models || !datas || !indices || !q0s || !actions || !output ||
        !errors || !completed || !warning_counts || !warning_lastinfo)
      return fail(error,error_length,"Null native PPO pointer");
    if (mj_version()!=mjVERSION_HEADER || !no_callbacks())
      return fail(error,error_length,"Version mismatch or unexpected global callback");
    if (batch<1 || batch>4096 || threads<1 || threads>64 ||
        (stepping!=0 && stepping!=1) || output_length!=static_cast<size_t>(batch)*kOutput ||
        !finite(q0s,static_cast<size_t>(batch)*kNu) ||
        !finite(actions,static_cast<size_t>(batch)*kNu))
      return fail(error,error_length,"Invalid native PPO dimensions or values");
    // Validate every object before launching workers: no partial global rejection.
    for (int b=0; b<batch; ++b) {
      const mjModel* m=models[b]; const mjData* d=datas[b];
      if (!m || !d || m->nq!=kNq || m->nv!=kNv || m->nu!=kNu ||
          m->na || m->nmocap || m->nuserdata || m->nplugin || m->nhistory ||
          m->nflex || m->neq || m->ntendon || d->threadpool ||
          m->opt.integrator!=mjINT_IMPLICITFAST || m->opt.solver!=mjSOL_NEWTON ||
          std::abs(m->opt.timestep-.005)>1e-15)
        return fail(error,error_length,"Unsupported PPO model or nested MuJoCo threadpool");
      const int* ids=indices+b*kIds;
      for (int j=0; j<kNu; ++j)
        if (ids[j]<7 || ids[j]>=kNq || ids[kNu+j]<6 || ids[kNu+j]>=kNv ||
            ids[2*kNu+j]<0 || ids[2*kNu+j]>=kNu)
          return fail(error,error_length,"Invalid native PPO joint index");
      for (int j=0; j<kBodies; ++j)
        if (ids[3*kNu+j]<=0 || ids[3*kNu+j]>=m->nbody)
          return fail(error,error_length,"Invalid native PPO body index");
    }
    std::fill(output, output+output_length, 0.);
    // Independent model/data per row; never share a MuJoCo workspace.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(batch>1)
#endif
    for (int b=0; b<batch; ++b) {
      const mjModel* m=models[b]; mjData* d=datas[b];
      const int* ids=indices+b*kIds;
      double* row=output+static_cast<size_t>(b)*kOutput;
      completed[b]=0;
      if (!errors[b]) {
        if (warning(d)) errors[b]=1;
        else if (external_force(m,d)) errors[b]=3;
        else if (!finite(d->qpos,kNq) || !finite(d->qvel,kNv) || !std::isfinite(d->time)) errors[b]=4;
      }
      const double initial_time=d->time;
      if (stepping && !errors[b]) {
        for (int sub=0; sub<kSubsteps; ++sub) {
          double* applied=row+kSnapshot+kSubsteps*kBodies*3+sub*kNu;
          torque(d,ids,q0s+b*kNu,actions+b*kNu,applied);
          for (int j=0; j<kNu; ++j) d->ctrl[ids[2*kNu+j]]=applied[j];
          // Persistent state, solver warm starts, and clock are retained.
          mj_step(m,d);
          if (warning(d)) { errors[b]=2; break; }
          mj_forward(m,d);
          if (warning(d)) { errors[b]=2; break; }
          contacts(m,d,ids,row+kSnapshot+sub*kBodies*3);
          completed[b]=sub+1;
          if (!finite(d->qpos,kNq) || !finite(d->qvel,kNv)) { errors[b]=4; break; }
          if (std::abs(d->time-initial_time-(sub+1)*.005)>1e-10) { errors[b]=5; break; }
          if (external_force(m,d)) { errors[b]=3; break; }
        }
      }
      snapshot(m,d,ids,row);
      if (!errors[b] && !finite(row,kOutput)) errors[b]=4;
      for (int i=0; i<mjNWARNING; ++i) {
        warning_counts[b*mjNWARNING+i]=d->warning[i].number;
        warning_lastinfo[b*mjNWARNING+i]=d->warning[i].lastinfo;
      }
    }
    return 0;
  } catch (const std::exception& exception) {
    return fail(error,error_length,exception.what());
  } catch (...) {
    return fail(error,error_length,"Unexpected native PPO C++ exception");
  }
}
}  // extern "C"
