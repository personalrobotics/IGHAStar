// UR5e + Robotiq peg insertion on CPU MuJoCo, for IGHA* through mujoco_env.h. C++ port of peg_problem.py's planner
// callbacks (sample_controls, _resolve, dynamics with osc_targets: absolute, validity, heuristic, goal_mask).
// UNREVIEWED: written by Claude, not yet read or reviewed by a human.
//
// Node state = PegProblem's layout exactly (env-origin-relative, quaternions xyzw), so configs, paths and the Isaac
// replay are shared with the Python backends:
//     [ hand xyz | hand rot6d | peg xyz | peg z axis | grip | grip_cmd | joint_pos (12) | joint_vel (12) |
//       peg pose (xyz, xyzw) | peg vel (lin world, ang world) ]
// MuJoCo side (verified against MuJoCo-Warp's state after a write, export_cpp_problem.py): joint order = Isaac's;
// world = env-relative + env origin; free-joint quaternion wxyz; free-joint angular velocity in the body frame.
// The hole, table and robot support are mocap bodies (static), and so is the robot base.
//
// Control [9]: dx dy dz (m / policy step, base frame) | rotation vector (rad / step, base frame) | grip (<0 close) |
//              kind (0 random, 1 toward a point of the peg, 2 toward the hole) | seed of the kind-1/2 target point
#pragma once

#include <mujoco_env.h>

#include <array>
#include <cmath>
#include <limits>

struct PegTask {
  // ---- layout
  int F = 16, o_grip = 16, o_q = 17, o_qd = 29, o_peg = 41, o_pegv = 48, n = 54, J = 12;
  std::vector<int> jq, jd;  // MuJoCo qpos / qvel address of Isaac joint i
  int peg_q = -1, peg_d = -1, wrist = -1, hand_body = -1;
  // ---- frames (env-relative; quaternions stored wxyz for mju_*)
  mjtNum origin[3]{}, base_p[3]{}, base_q[4]{1, 0, 0, 0}, hole_p[3]{}, hole_q[4]{1, 0, 0, 0};
  mjtNum ee_p[3]{}, ee_q[4]{1, 0, 0, 0}, pego_p[3]{}, pego_q[4]{1, 0, 0, 0}, holeo_p[3]{}, holeo_q[4]{1, 0, 0, 0};
  mjtNum ha_p[3]{}, ha_q[4]{1, 0, 0, 0};  // hole alignment frame = hole pose (+) hole offset
  mjtNum box_lo[3]{}, box_hi[3]{};
  std::vector<int> mocap_id;               // mocap ids, with their (env-relative) poses below
  std::vector<std::array<mjtNum, 7>> mocap_pose;  // pos, quat wxyz (world)
  // ---- controller / timing
  mjtNum kp[6]{}, kd[6]{}, tmax[6]{}, lever[3]{};
  int T = 5, nsub = 12;
  double step_dt = 0.1, edge_dt = 0.5, grip_open = 0.0, grip_close = 0.785398;
  // ---- sampler
  float d_xyz[3]{}, d_rot[3]{};
  int K = 64, n_peg_rows = 8, n_hole_rows = 8;
  double peg_grasped_dist = 0.06, hole_jitter = 0.01, hole_height = 0.04;
  // ---- validity / goal / heuristic
  double max_joint_vel = 20, max_peg_accel = 1.1 * 9.81, table_top_z = 0, table_z_margin = 0.05;
  double pos_thr = 0.0025, rot_thr = 0.025, max_object_step = 0.17, max_rot_step = 0.87;

  struct Scratch {
    std::vector<mjtNum> jacp, jacr;
    mjtNum pt[16][3], qt[16][4];  // per policy step: OSC target pose of the wrist body (world)
    mjtNum prev_v[3];
    float cmd = 1.f;
  };

  static void xyzw_to_wxyz(const double *a, mjtNum *q) { q[0] = a[3], q[1] = a[0], q[2] = a[1], q[3] = a[2]; }

  void init(const py::dict &mj, const mjModel *m) {
    using mjenv::dnum;
    using mjenv::dvec;
    auto ivec = [&](const char *k) {
      std::vector<int> o;
      for (double v : dvec(mj, k)) o.push_back((int)v);
      return o;
    };
    F = (int)dnum(mj, "F"), o_grip = (int)dnum(mj, "o_grip"), o_q = (int)dnum(mj, "o_q"), o_qd = (int)dnum(mj, "o_qd");
    o_peg = (int)dnum(mj, "o_peg"), o_pegv = (int)dnum(mj, "o_pegv"), n = (int)dnum(mj, "n"), J = (int)dnum(mj, "J");
    if (n != n_dims || n_cont != 9) throw std::runtime_error("peg_mujoco: compiled N_DIMS/N_CONT do not match the state layout");
    jq = ivec("joint_qadr"), jd = ivec("joint_dadr");
    if ((int)jq.size() != J || (int)jd.size() != J) throw std::runtime_error("peg_mujoco: joint maps need J entries");
    peg_q = (int)dnum(mj, "peg_qadr"), peg_d = (int)dnum(mj, "peg_dadr");
    wrist = (int)dnum(mj, "wrist_body"), hand_body = (int)dnum(mj, "hand_body");
    auto v3 = [&](const char *k, mjtNum *o) { auto v = dvec(mj, k); for (int i = 0; i < 3; ++i) o[i] = v[i]; };
    auto q4 = [&](const char *k, mjtNum *o) { auto v = dvec(mj, k); xyzw_to_wxyz(v.data(), o); };
    auto pose = [&](const char *k, mjtNum *p, mjtNum *q) { auto v = dvec(mj, k); for (int i = 0; i < 3; ++i) p[i] = v[i]; xyzw_to_wxyz(v.data() + 3, q); };
    v3("origin", origin);
    pose("base_pose", base_p, base_q);
    pose("hole_pose", hole_p, hole_q);
    v3("ee_offset_pos", ee_p), q4("ee_offset_quat", ee_q);
    v3("peg_offset_pos", pego_p), q4("peg_offset_quat", pego_q);
    v3("hole_offset_pos", holeo_p), q4("hole_offset_quat", holeo_q);
    combine(hole_p, hole_q, holeo_p, holeo_q, ha_p, ha_q);
    auto box = dvec(mj, "peg_local_box");
    for (int i = 0; i < 3; ++i) box_lo[i] = box[i], box_hi[i] = box[3 + i];
    // mocap bodies: ids + env-relative poses (xyzw) -> world, wxyz
    mocap_id = ivec("mocap_ids");
    auto mp = dvec(mj, "mocap_poses");
    if (mp.size() != 7 * mocap_id.size()) throw std::runtime_error("peg_mujoco: mocap_poses must be 7 per mocap id");
    for (size_t i = 0; i < mocap_id.size(); ++i) {
      std::array<mjtNum, 7> a;
      for (int k = 0; k < 3; ++k) a[k] = mp[7 * i + k] + origin[k];
      xyzw_to_wxyz(&mp[7 * i + 3], &a[3]);
      mocap_pose.push_back(a);
    }
    auto k6 = dvec(mj, "kp"), d6 = dvec(mj, "kd"), t6 = dvec(mj, "tau_max");
    for (int i = 0; i < 6; ++i) kp[i] = k6[i], kd[i] = d6[i], tmax[i] = t6[i];
    v3("lever", lever);
    T = (int)dnum(mj, "T"), nsub = (int)dnum(mj, "decimation");
    if (T > 16) throw std::runtime_error("peg_mujoco: at most 16 policy steps per edge");
    step_dt = dnum(mj, "step_dt"), edge_dt = dnum(mj, "edge_dt");
    grip_open = dnum(mj, "grip_open", 0.0), grip_close = dnum(mj, "grip_close", 0.785398);
    auto dx = dvec(mj, "d_xyz"), dr = dvec(mj, "d_rot");
    for (int i = 0; i < 3; ++i) d_xyz[i] = (float)dx[i], d_rot[i] = (float)dr[i];
    K = (int)dnum(mj, "num_controls"), n_peg_rows = (int)dnum(mj, "n_peg_rows"), n_hole_rows = (int)dnum(mj, "n_hole_rows");
    peg_grasped_dist = dnum(mj, "peg_grasped_dist"), hole_jitter = dnum(mj, "hole_jitter"), hole_height = dnum(mj, "hole_height");
    max_joint_vel = dnum(mj, "max_joint_vel"), max_peg_accel = dnum(mj, "max_peg_accel_g") * 9.81;
    table_top_z = dnum(mj, "table_top_z"), table_z_margin = dnum(mj, "table_z_margin");
    pos_thr = dnum(mj, "pos_thr"), rot_thr = dnum(mj, "rot_thr");
    max_object_step = dnum(mj, "max_object_step"), max_rot_step = dnum(mj, "max_rot_step");
    if (std::abs(m->opt.timestep * nsub - step_dt) > 1e-6)
      throw std::runtime_error("peg_mujoco: model timestep x decimation != step_dt");
  }
  void make_scratch(const mjModel *m, Scratch &s) const { s.jacp.resize(3 * m->nv), s.jacr.resize(3 * m->nv); }
  int policy_steps() const { return T; }
  int substeps() const { return nsub; }
  int num_controls() const { return K; }

  // ---------------------------------------------------------------- geometry helpers (wxyz)
  static void combine(const mjtNum *p1, const mjtNum *q1, const mjtNum *p2, const mjtNum *q2, mjtNum *p, mjtNum *q) {
    mjtNum r[3];
    mju_rotVecQuat(r, p2, q1);
    mju_add3(p, p1, r);
    mju_mulQuat(q, q1, q2);
  }
  // rotation vector (in the base frame) -> world-frame quaternion
  void rotvec_world(const mjtNum *rv_b, mjtNum *dq) const {
    mjtNum rv[3];
    mju_rotVecQuat(rv, rv_b, base_q);
    const mjtNum a = mju_norm3(rv);
    dq[0] = 1, dq[1] = dq[2] = dq[3] = 0;
    if (a > 1e-9) {
      mjtNum ax[3] = {rv[0] / a, rv[1] / a, rv[2] / a};
      mju_axisAngle2Quat(dq, ax, a);
    }
  }
  // OmniReset ProgressContext: distance and |roll|+|pitch| of the peg alignment frame in the hole alignment frame
  void alignment(const float *x, double &dist, double &ang) const {
    mjtNum pp[3] = {x[o_peg], x[o_peg + 1], x[o_peg + 2]}, pq[4], pa_p[3], pa_q[4];
    const double qx[4] = {x[o_peg + 3], x[o_peg + 4], x[o_peg + 5], x[o_peg + 6]};
    xyzw_to_wxyz(qx, pq);
    mju_normalize4(pq);
    combine(pp, pq, pego_p, pego_q, pa_p, pa_q);
    mjtNum d[3], rel_p[3], hinv[4], rel_q[4];
    mju_sub3(d, pa_p, ha_p);
    mju_negQuat(hinv, ha_q);
    mju_rotVecQuat(rel_p, d, hinv);
    mju_mulQuat(rel_q, hinv, pa_q);
    const double w = rel_q[0], X = rel_q[1], Y = rel_q[2], Z = rel_q[3];
    const double roll = std::atan2(2.0 * (w * X + Y * Z), 1 - 2 * (X * X + Y * Y));
    const double sp = 2.0 * (w * Y - Z * X);
    const double pitch = std::abs(sp) >= 1 ? std::copysign(M_PI / 2, sp) : std::asin(sp);
    dist = mju_norm3(rel_p);
    ang = std::abs(roll) + std::abs(pitch);
  }

  // ---------------------------------------------------------------- sampler (= PegProblem.sample_controls)
  void sample_controls(std::mt19937_64 &rng, int Kc, float *u) {
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::uniform_int_distribution<int> seed(0, (1 << 24) - 1);  // exact in float32
    for (int k = 0; k < Kc; ++k) {
      float *r = u + (size_t)k * 9;
      for (int i = 0; i < 7; ++i) r[i] = U(rng);
      for (int i = 0; i < 3; ++i) r[i] *= d_xyz[i], r[3 + i] *= d_rot[i];
      r[7] = 0.f, r[8] = 0.f;
      if (k >= Kc - n_peg_rows - n_hole_rows) {
        r[7] = k >= Kc - n_hole_rows ? 2.f : 1.f;
        r[8] = (float)seed(rng);
      }
    }
  }

  // ---------------------------------------------------------------- one edge
  bool begin(const mjModel *m, mjData *d, const float *x, const float *u, Scratch &s) const {
    for (int i = 0; i < n; ++i)
      if (!std::isfinite(x[i])) return false;
    mj_resetData(m, d);
    for (int i = 0; i < J; ++i) d->qpos[jq[i]] = x[o_q + i], d->qvel[jd[i]] = x[o_qd + i];
    mjtNum pq[4];
    const double qx[4] = {x[o_peg + 3], x[o_peg + 4], x[o_peg + 5], x[o_peg + 6]};
    xyzw_to_wxyz(qx, pq);
    mju_normalize4(pq);
    for (int i = 0; i < 3; ++i) d->qpos[peg_q + i] = x[o_peg + i] + origin[i];
    mju_copy4(d->qpos + peg_q + 3, pq);
    mjtNum w_world[3] = {x[o_pegv + 3], x[o_pegv + 4], x[o_pegv + 5]}, w_local[3], pinv[4];
    mju_negQuat(pinv, pq);
    mju_rotVecQuat(w_local, w_world, pinv);
    for (int i = 0; i < 3; ++i) d->qvel[peg_d + i] = x[o_pegv + i], d->qvel[peg_d + 3 + i] = w_local[i];
    for (size_t i = 0; i < mocap_id.size(); ++i) {
      mju_copy3(d->mocap_pos + 3 * mocap_id[i], mocap_pose[i].data());
      mju_copy4(d->mocap_quat + 4 * mocap_id[i], mocap_pose[i].data() + 3);
    }
    // ---- resolve the control (= PegProblem._resolve)
    mjtNum a[6];
    for (int i = 0; i < 6; ++i) a[i] = u[i];
    s.cmd = u[6] < 0 ? -1.f : 1.f;
    const int kind = (int)u[7];
    if (kind > 0) {
      std::mt19937 g((uint32_t)u[8]);
      std::uniform_real_distribution<double> R01(0.0, 1.0);
      const double rnd[3] = {R01(g), R01(g), R01(g)};
      mjtNum peg_p[3] = {x[o_peg], x[o_peg + 1], x[o_peg + 2]}, target[3], ref[3];
      if (kind == 1) {
        mjtNum pl[3], pw[3];
        for (int i = 0; i < 3; ++i) pl[i] = box_lo[i] + rnd[i] * (box_hi[i] - box_lo[i]);
        mju_rotVecQuat(pw, pl, pq);
        mju_add3(target, pw, peg_p);
      } else {
        target[0] = ha_p[0] + (rnd[0] - 0.5) * 2 * hole_jitter;
        target[1] = ha_p[1] + (rnd[1] - 0.5) * 2 * hole_jitter;
        target[2] = ha_p[2] + rnd[2] * hole_height;
      }
      const mjtNum hand[3] = {x[0], x[1], x[2]};
      mjtNum dh[3];
      mju_sub3(dh, hand, peg_p);
      const bool held = x[o_grip] < 0 && mju_norm3(dh) < peg_grasped_dist;
      if (held) {
        mjtNum pa_p[3], pa_q[4];
        combine(peg_p, pq, pego_p, pego_q, pa_p, pa_q);
        mju_copy3(ref, pa_p);
      } else {
        mju_copy3(ref, hand);
      }
      mjtNum dw[3], db[3], binv[4];
      mju_sub3(dw, target, ref);
      mju_negQuat(binv, base_q);
      mju_rotVecQuat(db, dw, binv);
      for (int i = 0; i < 3; ++i) a[i] = mju_clip(db[i] / T, -d_xyz[i], d_xyz[i]);
    }
    if (m->nu > 0) d->ctrl[0] = s.cmd < 0 ? grip_close : grip_open;
    if (m->nu > 1) d->ctrl[1] = 0.0;
    mj_forward(m, d);
    // ---- absolute OSC targets (= PegProblem._wrist_action_absolute): hand point + k dp, hand rotated by k rv
    mjtNum qw0[4], r0[3], ph0[3], dp_w[3];
    mju_copy4(qw0, d->xquat + 4 * wrist);
    mju_rotVecQuat(r0, lever, qw0);
    mju_add3(ph0, d->xpos + 3 * wrist, r0);
    mju_rotVecQuat(dp_w, a, base_q);
    for (int t = 0; t < T; ++t) {
      const mjtNum k = t + 1;
      mjtNum rv[3] = {k * a[3], k * a[4], k * a[5]}, dq[4], rt[3];
      rotvec_world(rv, dq);
      mju_mulQuat(s.qt[t], dq, qw0);
      mju_rotVecQuat(rt, lever, s.qt[t]);
      for (int i = 0; i < 3; ++i) s.pt[t][i] = ph0[i] + k * dp_w[i] - rt[i];
    }
    for (int i = 0; i < 3; ++i) s.prev_v[i] = x[o_pegv + i];
    return true;
  }

  // UWLab RelCartesianOSC: tau = clamp(J^T (Kp [p_t - p; axisangle(q_t q^-1)] - Kd J qd), +-tau_max), arm dofs 0..5
  void control(const mjModel *m, mjData *d, int t, int sub, Scratch &s) const {
    (void)sub;
    const int nv = m->nv;
    mj_jacBody(m, d, s.jacp.data(), s.jacr.data(), wrist);
    const mjtNum *xp = d->xpos + 3 * wrist;
    mjtNum qc[4], qe[4], err[6], F6[6];
    mju_negQuat(qc, d->xquat + 4 * wrist);
    mju_mulQuat(qe, s.qt[t], qc);
    for (int i = 0; i < 3; ++i) err[i] = s.pt[t][i] - xp[i];
    mju_quat2Vel(err + 3, qe, 1.0);
    for (int i = 0; i < 6; ++i) {
      const mjtNum *row = i < 3 ? s.jacp.data() + i * nv : s.jacr.data() + (i - 3) * nv;
      mjtNum v = 0;
      for (int j = 0; j < 6; ++j) v += row[j] * d->qvel[jd[j]];
      F6[i] = kp[i] * err[i] - kd[i] * v;
    }
    mju_zero(d->qfrc_applied, nv);
    for (int j = 0; j < 6; ++j) {
      const int c = jd[j];
      mjtNum tau = 0;
      for (int i = 0; i < 6; ++i) tau += (i < 3 ? s.jacp[i * nv + c] : s.jacr[(i - 3) * nv + c]) * F6[i];
      d->qfrc_applied[c] = mju_clip(tau, -tmax[j], tmax[j]);
    }
  }

  // state after policy step t (= PegProblem._read_states) + per-step validity (joint speed, peg acceleration, finite)
  bool record(const mjModel *m, mjData *d, int t, Scratch &s, float *o) const {
    (void)t;
    mj_kinematics(m, d);
    encode(m, d, s.cmd, o);
    bool ok = true;
    for (int i = 0; i < n; ++i) ok = ok && std::isfinite(o[i]);
    for (int i = 0; i < J; ++i) ok = ok && std::abs(o[o_qd + i]) <= max_joint_vel;
    mjtNum dv[3] = {o[o_pegv] - s.prev_v[0], o[o_pegv + 1] - s.prev_v[1], o[o_pegv + 2] - s.prev_v[2]};
    ok = ok && mju_norm3(dv) / step_dt <= max_peg_accel;
    for (int i = 0; i < 3; ++i) s.prev_v[i] = o[o_pegv + i];
    return ok;
  }

  // pack d (kinematics up to date) into the node layout
  void encode(const mjModel *m, const mjData *d, float cmd, float *o) const {
    (void)m;
    mjtNum hp[3], hq[4], R[9];
    combine(d->xpos + 3 * hand_body, d->xquat + 4 * hand_body, ee_p, ee_q, hp, hq);
    mju_quat2Mat(R, hq);
    for (int i = 0; i < 3; ++i) o[i] = (float)(hp[i] - origin[i]);
    for (int i = 0; i < 3; ++i) o[3 + i] = (float)R[3 * i], o[6 + i] = (float)R[3 * i + 1];
    const mjtNum *pq = d->qpos + peg_q + 3;
    mjtNum Rp[9];
    mju_quat2Mat(Rp, pq);
    for (int i = 0; i < 3; ++i) o[9 + i] = (float)(d->qpos[peg_q + i] - origin[i]), o[12 + i] = (float)Rp[3 * i + 2];
    o[15] = cmd;
    o[o_grip] = cmd;
    for (int i = 0; i < J; ++i) o[o_q + i] = (float)d->qpos[jq[i]], o[o_qd + i] = (float)d->qvel[jd[i]];
    for (int i = 0; i < 3; ++i) o[o_peg + i] = o[9 + i];
    o[o_peg + 3] = (float)pq[1], o[o_peg + 4] = (float)pq[2], o[o_peg + 5] = (float)pq[3], o[o_peg + 6] = (float)pq[0];
    mjtNum wl[3] = {d->qvel[peg_d + 3], d->qvel[peg_d + 4], d->qvel[peg_d + 5]}, ww[3];
    mju_rotVecQuat(ww, wl, pq);
    for (int i = 0; i < 3; ++i) o[o_pegv + i] = (float)d->qvel[peg_d + i], o[o_pegv + 3 + i] = (float)ww[i];
  }

  bool state_valid(const float *x) const {
    for (int i = 0; i < n; ++i)
      if (!std::isfinite(x[i])) return false;
    return x[o_peg + 2] > table_top_z - table_z_margin;
  }
  float cost(const float *, const float *, const float *) const { return (float)edge_dt; }
  bool goal(const float *x) const {
    double dist, ang;
    alignment(x, dist, ang);
    return dist < pos_thr && ang < rot_thr;
  }
  float heuristic(const float *x) const {
    double dist, ang;
    alignment(x, dist, ang);
    const double gap = std::max(dist - pos_thr, 0.0);
    const double v = std::sqrt(double(x[o_pegv]) * x[o_pegv] + double(x[o_pegv + 1]) * x[o_pegv + 1] +
                               double(x[o_pegv + 2]) * x[o_pegv + 2]) * edge_dt;
    const double h_pos = gap / std::max(v, max_object_step) * edge_dt;
    const double h_rot = std::max(ang - rot_thr, 0.0) / (2.0 * max_rot_step) * edge_dt;
    return (float)std::max(h_pos, h_rot);
  }
};

class Environment : public MujocoEnvironment<PegTask> {
 public:
  using MujocoEnvironment<PegTask>::MujocoEnvironment;
};
