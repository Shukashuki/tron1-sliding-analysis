// Optional, single-worker exact-transition prototype. No callbacks are installed.
// MuJoCo fatal errors are intentionally contained by the Python-owned subprocess,
// NOT by a global error handler or unsupported private thread-local symbols.
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <limits>

static_assert(mjVERSION_HEADER == 3008001, "Requires MuJoCo 3.8.1 headers");
static_assert(sizeof(mjtNum) == sizeof(double), "Requires double precision MuJoCo");

namespace {
constexpr int kNq = 15, kNv = 14, kNu = 8, kBodies = 9, kSubsteps = 4;
constexpr int kHistory = 3 * kBodies * 3;
constexpr int kBodyState = kBodies * (3 + 4 + 3 + 3);
constexpr int kPre = kNq + kNv + kBodyState;
constexpr int kStep = kNq + kNv + kNu + kBodyState + kBodies * 3 + 2;
constexpr int kStage = kPre + kSubsteps * kStep + kHistory;

int fail(char* error, size_t length, const char* message) {
  if (error && length) std::snprintf(error, length, "%s", message);
  return 1;
}

bool finite(const double* value, size_t count) {
  for (size_t i = 0; i < count; ++i) if (!std::isfinite(value[i])) return false;
  return true;
}

bool no_callbacks() {
  return !(mjcb_control || mjcb_passive || mjcb_sensor || mjcb_contactfilter ||
           mjcb_act_dyn || mjcb_act_gain || mjcb_act_bias);
}

// Matches the existing NumPy DC-motor helper, not MuJoCo's different DC model.
void torque(const mjData* d, const int* ids, const double* q0,
            const double* action, double* out) {
  for (int j = 0; j < kNu; ++j) {
    const double a = std::clamp(action[j], -1.0, 1.0);
    const double dq = d->qvel[ids[kNu + j]];
    const double effort = j < 6 ? 80.0 : 12.0;
    const double speed = j < 6 ? 15.0 : 100.0;
    const double requested = j < 6 ?
        500.0 * (q0[j] + a - d->qpos[ids[j]]) - 10.0 * dq : 12.0 * a;
    const double v = std::clamp(dq, -2.0 * speed, 2.0 * speed);
    const double lower = std::max(effort * (-1.0 - v / speed), -effort);
    const double upper = std::min(effort * (1.0 - v / speed), effort);
    out[j] = std::clamp(requested, lower, upper);
  }
}

void body_state(const mjModel* m, const mjData* d, const int* bids, double* out) {
  double* pos = out;
  double* quat = pos + kBodies * 3;
  double* linear = quat + kBodies * 4;
  double* angular = linear + kBodies * 3;
  for (int i = 0; i < kBodies; ++i) {
    const int body = bids[i];
    std::memcpy(pos + 3*i, d->xpos + 3*body, 3*sizeof(double));
    std::memcpy(quat + 4*i, d->xquat + 4*body, 4*sizeof(double));
    double velocity[6];
    mj_objectVelocity(m, d, mjOBJ_BODY, body, velocity, 0);
    std::memcpy(linear + 3*i, velocity + 3, 3*sizeof(double));
    std::memcpy(angular + 3*i, velocity, 3*sizeof(double));
  }
}

void contacts(const mjModel* m, const mjData* d, const int* bids, double* out) {
  std::fill(out, out + kBodies * 3, 0.0);
  for (int c = 0; c < d->ncon; ++c) {
    double wrench[6], world[3];
    mj_contactForce(m, d, c, wrench);
    const double* frame = d->contact[c].frame;
    for (int axis = 0; axis < 3; ++axis) {
      world[axis] = frame[axis] * wrench[0] + frame[3 + axis] * wrench[1] + frame[6 + axis] * wrench[2];
    }
    for (int side = 0; side < 2; ++side) {
      const int geom = d->contact[c].geom[side];
      if (geom < 0) continue;  // Flex models are rejected; defensive only.
      const int body = m->geom_bodyid[geom];
      const double sign = side ? 1.0 : -1.0;
      for (int i = 0; i < kBodies; ++i) if (bids[i] == body) {
        for (int axis = 0; axis < 3; ++axis) out[3*i + axis] += sign * world[axis];
      }
    }
  }
}

bool warnings(const mjData* d) {
  for (int i = 0; i < mjNWARNING; ++i) if (d->warning[i].number) return true;
  return false;
}
}  // namespace

extern "C" {
int tron1_batch_abi() noexcept { return 1; }
int tron1_header_version() noexcept { return mjVERSION_HEADER; }
int tron1_runtime_version() noexcept { return mj_version(); }
int tron1_real_size() noexcept { return sizeof(mjtNum); }
int tron1_stage_width() noexcept { return kStage; }
const char* tron1_linked_mujoco_path() noexcept {
  Dl_info info;
  return dladdr(reinterpret_cast<void*>(mj_version), &info) ? info.dli_fname : nullptr;
}

// Test-only fault injection, called exclusively inside the isolated worker.
void tron1_native_error_probe() { mju_error("Intentional native error containment probe"); }

// ids: qpos[8], qvel[8], actuator[8], body[9], all in logical action/body order.
// Output per stage: pre(qpos,qvel,body state), 4 substeps(qpos,qvel,torque,
// body state,net contact forces,interval time,ncon), latest 3 contact samples.
int tron1_batch(const mjModel* m, mjData* d, mjData* measurement,
                int batch, int horizon, const int* ids, const double* q0,
                const double* initial_qpos, const double* initial_qvel,
                const double* initial_history, const double* actions,
                double* output, size_t output_length, char* error, size_t error_length) noexcept {
  try {
    if (!m || !d || !measurement || d == measurement || !ids || !q0 ||
        !initial_qpos || !initial_qvel || !initial_history || !actions || !output)
      return fail(error, error_length, "Null pointer or aliased native workspaces");
    if (mj_version() != mjVERSION_HEADER || !no_callbacks())
      return fail(error, error_length, "MuJoCo version mismatch or unexpected global callback");
    if (batch < 1 || batch > 4096 || horizon < 1 || horizon > 1024 ||
        output_length != static_cast<size_t>(batch) * horizon * kStage)
      return fail(error, error_length, "Invalid native batch dimensions/output length");
    if (m->nq != kNq || m->nv != kNv || m->nu != kNu || m->na || m->nmocap ||
        m->nuserdata || m->nplugin || m->nhistory || m->nflex || m->neq || m->ntendon ||
        m->opt.integrator != mjINT_IMPLICITFAST || m->opt.solver != mjSOL_NEWTON ||
        std::abs(m->opt.timestep - 0.005) > 1e-15)
      return fail(error, error_length, "Unsupported model size, hidden state or integration contract");
    if (!finite(q0, kNu) || !finite(initial_qpos, static_cast<size_t>(batch)*kNq) ||
        !finite(initial_qvel, static_cast<size_t>(batch)*kNv) ||
        !finite(initial_history, static_cast<size_t>(batch)*kHistory) ||
        !finite(actions, static_cast<size_t>(batch)*horizon*kNu))
      return fail(error, error_length, "Nonfinite native input");
    for (size_t i = 0; i < static_cast<size_t>(batch)*horizon*kNu; ++i)
      if (std::abs(actions[i]) > 1.0 + 1e-9) return fail(error, error_length, "Unbounded normalized action");
    for (int j = 0; j < kNu; ++j) {
      const int qp = ids[j], v = ids[kNu+j], actuator = ids[2*kNu+j];
      if (qp < 7 || qp >= kNq || v < 6 || v >= kNv || actuator < 0 || actuator >= kNu)
        return fail(error, error_length, "Native joint/actuator index outside model");
      for (int i = 0; i < j; ++i) if (ids[i] == qp || ids[kNu+i] == v || ids[2*kNu+i] == actuator)
        return fail(error, error_length, "Duplicate native joint/actuator index");
      const int joint = m->actuator_trnid[2*actuator];
      if (m->actuator_trntype[actuator] != mjTRN_JOINT || joint < 0 || joint >= m->njnt ||
          m->jnt_type[joint] != mjJNT_HINGE || m->jnt_qposadr[joint] != qp || m->jnt_dofadr[joint] != v)
        return fail(error, error_length, "Actuator/joint transmission mismatch");
    }
    const int* bids = ids + 3*kNu;
    for (int i = 0; i < kBodies; ++i) {
      if (bids[i] <= 0 || bids[i] >= m->nbody) return fail(error, error_length, "Native body index outside model");
      for (int j = 0; j < i; ++j) if (bids[i] == bids[j]) return fail(error, error_length, "Duplicate native body index");
    }
    for (int b = 0; b < batch; ++b) {
      double q[kNq], v[kNv], history[kHistory];
      std::memcpy(q, initial_qpos + b*kNq, sizeof(q));
      std::memcpy(v, initial_qvel + b*kNv, sizeof(v));
      std::memcpy(history, initial_history + b*kHistory, sizeof(history));
      for (int h = 0; h < horizon; ++h) {
        double* stage = output + (static_cast<size_t>(b)*horizon + h)*kStage;
        std::memcpy(stage, q, sizeof(q));
        std::memcpy(stage + kNq, v, sizeof(v));
        // Match legacy measure(): separate cold zero-control workspace.
        mj_resetData(m, measurement);
        std::memcpy(measurement->qpos, q, sizeof(q));
        std::memcpy(measurement->qvel, v, sizeof(v));
        mj_forward(m, measurement);
        if (warnings(measurement)) return fail(error, error_length, "MuJoCo warning during pre-stage measurement");
        body_state(m, measurement, bids, stage + kNq + kNv);
        // Exact cold reset at EACH 20 ms interval, not only the horizon start.
        mj_resetData(m, d);
        std::memcpy(d->qpos, q, sizeof(q));
        std::memcpy(d->qvel, v, sizeof(v));
        const double* action = actions + (static_cast<size_t>(b)*horizon + h)*kNu;
        for (int sub = 0; sub < kSubsteps; ++sub) {
          double applied[kNu];
          torque(d, ids, q0, action, applied);
          for (int j = 0; j < kNu; ++j) d->ctrl[ids[2*kNu+j]] = applied[j];
          mj_step(m, d);
          if (warnings(d)) return fail(error, error_length, "MuJoCo warning during native physics step");
          // Do not recompute torque before this post-integration forward.
          mj_forward(m, d);
          if (warnings(d)) return fail(error, error_length, "MuJoCo warning during post-step forward");
          double* sample = stage + kPre + sub*kStep;
          std::memcpy(sample, d->qpos, sizeof(q));
          std::memcpy(sample + kNq, d->qvel, sizeof(v));
          std::memcpy(sample + kNq + kNv, applied, sizeof(applied));
          body_state(m, d, bids, sample + kNq + kNv + kNu);
          double* force = sample + kNq + kNv + kNu + kBodyState;
          contacts(m, d, bids, force);
          std::memmove(history, history + kBodies*3, (kHistory - kBodies*3)*sizeof(double));
          std::memcpy(history + kHistory - kBodies*3, force, kBodies*3*sizeof(double));
          sample[kStep-2] = d->time;
          sample[kStep-1] = d->ncon;
          if (!finite(sample, kStep) || std::abs(d->time - (sub+1)*0.005) > 1e-10)
            return fail(error, error_length, "Nonfinite output or hidden native clock reset");
          for (int j = 0; j < kNv; ++j) if (d->qfrc_applied[j] != 0)
            return fail(error, error_length, "Unexpected external generalized force");
          for (int j = 0; j < 6*m->nbody; ++j) if (d->xfrc_applied[j] != 0)
            return fail(error, error_length, "Unexpected external body force");
        }
        std::memcpy(stage + kPre + kSubsteps*kStep, history, sizeof(history));
        std::memcpy(q, d->qpos, sizeof(q));
        std::memcpy(v, d->qvel, sizeof(v));
      }
    }
    return 0;
  } catch (const std::exception& exception) {
    return fail(error, error_length, exception.what());
  } catch (...) {
    return fail(error, error_length, "Unexpected native C++ exception");
  }
}
}  // extern "C"
