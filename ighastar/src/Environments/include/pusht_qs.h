// Mason / Lynch quasi-static PushT (ellipsoidal limit surface).
// Same state/action layout as pusht_physics.h, but no Chipmunk — pure geometry + LS.
//
// Model (Coulomb frictional point contact + ellipsoidal support limit surface):
//   LS = limit surface of support friction under the T (not pusher friction).
//   H = diag(1, 1, 1/ℓ²),  ℓ = τ_max/f_max ≈ (2/3) R_support  (Howe/Cutkosky)
//   Contact force on T lies in the friction cone about inward normal −n
//   (μ = Chipmunk agent–tee fa·fb): fdir = normalize(−n + clamp(vt/vn,±μ) t).
//   Unit wrench ĝ = [fdir; r × fdir]; associated flow ξ = λ H ĝ
//   maintain contact: ĝ·ξ = v_in  ⇒  λ = max(0, fdir·v_agent) / (ĝ·Hĝ)
//
// Succ action rings (host): when contacting, half of n_directions lie in the
// Coulomb friction cone about the inward normal (μ from Chipmunk shape pair),
// half in the outward 180° hemisphere (leave the surface). Agent PD targets
// are world-frame offsets along those angles.
//
// Not gym-pusht parity — use for A/B vs Chipmunk Succ.
#pragma once

#include <cmath>
#include <cstring>
#include <vector>

#ifndef PUSHT_QS_HD
#ifdef __CUDACC__
#define PUSHT_QS_HD __host__ __device__ __forceinline__
#else
#define PUSHT_QS_HD inline
#endif
#endif

namespace pusht_qs {

constexpr float kDt = 0.01f;
constexpr float kP = 100.f;
constexpr float kV = 20.f;
constexpr float kAgentRadius = 15.f;
constexpr float kTeeScale = 30.f;
constexpr float kTeeLength = 4.f;
constexpr int kStateDim = 10;
constexpr int kDefaultSubsteps = 10;
constexpr float kPi = 3.14159265358979323846f;

// Equivalent circular support radius for ellipsoidal LS (px). Default ~ tee half-span.
constexpr float kDefaultSupportRadius = 60.f;
// Coulomb μ for agent–tee = Chipmunk fa*fb with fa=fb=kShapeFriction(=1).
constexpr float kContactFrictionMu = 1.f;

// Device-safe float math (std:: variants are host-only under nvcc).
PUSHT_QS_HD float qs_sqrt(float x) { return sqrtf(x); }
PUSHT_QS_HD float qs_cos(float x) { return cosf(x); }
PUSHT_QS_HD float qs_sin(float x) { return sinf(x); }
PUSHT_QS_HD float qs_fmod(float x, float y) { return fmodf(x, y); }
PUSHT_QS_HD float qs_min(float a, float b) { return fminf(a, b); }
PUSHT_QS_HD float qs_max(float a, float b) { return fmaxf(a, b); }

struct Vec2 {
  float x, y;
};

PUSHT_QS_HD Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
PUSHT_QS_HD Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
PUSHT_QS_HD Vec2 operator*(float s, Vec2 a) { return {s * a.x, s * a.y}; }
PUSHT_QS_HD float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
PUSHT_QS_HD float cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; } // a×b scalar
PUSHT_QS_HD float norm(Vec2 a) { return qs_sqrt(dot(a, a)); }
PUSHT_QS_HD Vec2 normalized(Vec2 a) {
  float n = norm(a);
  return n > 1e-8f ? (1.f / n) * a : Vec2{1.f, 0.f};
}

PUSHT_QS_HD Vec2 rot(Vec2 v, float c, float s) {
  return {c * v.x - s * v.y, s * v.x + c * v.y};
}

// Local tee polys (same as Chipmunk / gym-pusht).
PUSHT_QS_HD void tee_cog_local(Vec2 &cog) {
  // centroids of the two rects, then average (matches pusht_physics.h)
  const float s = kTeeScale, L = kTeeLength;
  Vec2 c1{0.f, s * 0.5f};           // bar: x∈[-L s/2,L s/2], y∈[0,s]
  Vec2 c2{0.f, 0.5f * (s + L * s)}; // stem: x∈[-s/2,s/2], y∈[s, L s]
  cog = 0.5f * (c1 + c2);
}

// Axis-aligned rect in local frame: [xmin,xmax]×[ymin,ymax]
struct LocalRect {
  float xmin, xmax, ymin, ymax;
};

PUSHT_QS_HD void tee_rects(LocalRect &bar, LocalRect &stem) {
  const float s = kTeeScale, L = kTeeLength;
  bar = {-0.5f * L * s, 0.5f * L * s, 0.f, s};
  stem = {-0.5f * s, 0.5f * s, s, L * s};
}

// Closest point on AABB to q; returns penetration depth of circle (r) into rect
// (>0 if overlapping). Outward normal from rect toward circle center.
PUSHT_QS_HD bool circle_aabb(Vec2 q_local, float radius, const LocalRect &r,
                        float &pen, Vec2 &n_local, Vec2 &contact_local) {
  const float cx = qs_min(qs_max(q_local.x, r.xmin), r.xmax);
  const float cy = qs_min(qs_max(q_local.y, r.ymin), r.ymax);
  Vec2 closest{cx, cy};
  Vec2 d = q_local - closest;
  float dist = norm(d);
  if (dist > 1e-8f) {
    // circle center outside or on boundary of rect
    pen = radius - dist;
    if (pen <= 0.f)
      return false;
    n_local = normalized(d); // from rect toward center = outward from T
    contact_local = closest;
    return true;
  }
  // center inside rect: push out via nearest edge
  const float dl = q_local.x - r.xmin, dr = r.xmax - q_local.x;
  const float db = q_local.y - r.ymin, dt = r.ymax - q_local.y;
  const float m = qs_min(qs_min(dl, dr), qs_min(db, dt));
  if (m == dl) {
    n_local = {-1.f, 0.f};
    contact_local = {r.xmin, q_local.y};
  } else if (m == dr) {
    n_local = {1.f, 0.f};
    contact_local = {r.xmax, q_local.y};
  } else if (m == db) {
    n_local = {0.f, -1.f};
    contact_local = {q_local.x, r.ymin};
  } else {
    n_local = {0.f, 1.f};
    contact_local = {q_local.x, r.ymax};
  }
  pen = radius + m;
  return pen > 0.f;
}

// Deepest agent–tee contact. n_world: outward from T (toward agent).
PUSHT_QS_HD bool deepest_contact(Vec2 agent, Vec2 block_pos, float block_ang,
                            float &pen, Vec2 &n_world, Vec2 &contact_world,
                            Vec2 &cog_world) {
  const float c = qs_cos(block_ang), s = qs_sin(block_ang);
  Vec2 cog_l;
  tee_cog_local(cog_l);
  cog_world = block_pos + rot(cog_l, c, s);

  // agent center in block-local frame
  Vec2 d = agent - block_pos;
  Vec2 q_local{c * d.x + s * d.y, -s * d.x + c * d.y};

  LocalRect bar, stem;
  tee_rects(bar, stem);
  float pen0 = -1.f, pen1 = -1.f;
  Vec2 n0, n1, ct0, ct1;
  const bool h0 = circle_aabb(q_local, kAgentRadius, bar, pen0, n0, ct0);
  const bool h1 = circle_aabb(q_local, kAgentRadius, stem, pen1, n1, ct1);
  if (!h0 && !h1)
    return false;
  Vec2 n_l, ct_l;
  if (h0 && (!h1 || pen0 >= pen1)) {
    pen = pen0;
    n_l = n0;
    ct_l = ct0;
  } else {
    pen = pen1;
    n_l = n1;
    ct_l = ct1;
  }
  n_world = rot(n_l, c, s);
  contact_world = block_pos + rot(ct_l, c, s);
  return true;
}

PUSHT_QS_HD void clamp_speed(Vec2 &v, float max_speed) {
  if (max_speed <= 0.f)
    return;
  float sp = norm(v);
  if (sp > max_speed)
    v = (max_speed / sp) * v;
}

PUSHT_QS_HD void clamp_workspace(Vec2 &p, float lo, float hi) {
  p.x = qs_min(qs_max(p.x, lo), hi);
  p.y = qs_min(qs_max(p.y, lo), hi);
}

struct State {
  Vec2 agent, block;
  float ang;
  Vec2 agent_vel, block_vel;
  float omega;
};

PUSHT_QS_HD State from_array(const float *s) {
  return {{s[0], s[1]}, {s[2], s[3]}, s[4],
          {s[5], s[6]}, {s[7], s[8]}, s[9]};
}

PUSHT_QS_HD void to_array(const State &st, float *s) {
  s[0] = st.agent.x;
  s[1] = st.agent.y;
  s[2] = st.block.x;
  s[3] = st.block.y;
  float ang = qs_fmod(st.ang, 2.f * kPi);
  if (ang < 0)
    ang += 2.f * kPi;
  s[4] = ang;
  s[5] = st.agent_vel.x;
  s[6] = st.agent_vel.y;
  s[7] = st.block_vel.x;
  s[8] = st.block_vel.y;
  s[9] = st.omega;
}

// One QS substep: PD agent + ellipsoidal support-LS push on T if contacting.
// contact_mu: Coulomb μ at pusher–T (Chipmunk fa*fb). Support LS uses support_radius.
PUSHT_QS_HD float control_step(State &st, float ax, float ay, float max_agent_vel,
                          float support_radius, float workspace_lo,
                          float workspace_hi,
                          float contact_mu = kContactFrictionMu) {
  // Kinematic PD (same gains as Chipmunk path).
  Vec2 acc{kP * (ax - st.agent.x) + kV * (0.f - st.agent_vel.x),
           kP * (ay - st.agent.y) + kV * (0.f - st.agent_vel.y)};
  st.agent_vel = st.agent_vel + kDt * acc;
  clamp_speed(st.agent_vel, max_agent_vel);
  st.agent = st.agent + kDt * st.agent_vel;
  clamp_workspace(st.agent, workspace_lo, workspace_hi);

  float pen = 0.f;
  Vec2 n, contact, cog;
  if (!deepest_contact(st.agent, st.block, st.ang, pen, n, contact, cog)) {
    // Quasi-static: no inertia — block stops when not pushed.
    st.block_vel = {0.f, 0.f};
    st.omega = 0.f;
    return 0.f;
  }

  // n = outward from T. t = left-handed tangent. Unilateral: need vn > 0 into T.
  const Vec2 t{-n.y, n.x};
  const float vn = -dot(st.agent_vel, n); // agent speed into the T
  const float vt = dot(st.agent_vel, t);  // agent speed along the edge
  if (vn <= 1e-8f) {
    st.block_vel = {0.f, 0.f};
    st.omega = 0.f;
    return pen;
  }

  // Coulomb: force on T in cone about −n. ratio = vt/vn clamped to ±μ
  // (stick-like if |vt/vn|≤μ, else sliding on the cone boundary).
  const float mu = qs_max(contact_mu, 0.f);
  float ratio = vt / vn;
  if (ratio > mu)
    ratio = mu;
  else if (ratio < -mu)
    ratio = -mu;
  Vec2 fdir = normalized(Vec2{-n.x + ratio * t.x, -n.y + ratio * t.y});

  Vec2 r = contact - cog;
  const float tau_hat = cross(r, fdir); // r × f
  const float ell = (2.f / 3.f) * qs_max(support_radius, 1.f); // τmax/fmax
  const float ell2 = ell * ell;
  // ĝ·Hĝ = fx²+fy²+τ²/ℓ²  (H = diag(1,1,1/ℓ²))
  const float gHg =
      fdir.x * fdir.x + fdir.y * fdir.y + (tau_hat * tau_hat) / ell2;

  const float v_in = qs_max(0.f, dot(st.agent_vel, fdir));
  const float lambda = (gHg > 1e-12f) ? (v_in / gHg) : 0.f;

  // Twist ξ = λ H ĝ
  st.block_vel = {lambda * fdir.x, lambda * fdir.y};
  st.omega = lambda * tau_hat / ell2;
  // Integrate about CoG: origin = CoG − R cog_local.
  Vec2 cog_l;
  tee_cog_local(cog_l);
  const float c = qs_cos(st.ang), s = qs_sin(st.ang);
  Vec2 cog_w = st.block + rot(cog_l, c, s);
  cog_w = cog_w + kDt * st.block_vel;
  st.ang += kDt * st.omega;
  const float c2 = qs_cos(st.ang), s2 = qs_sin(st.ang);
  st.block = cog_w - rot(cog_l, c2, s2);
  clamp_workspace(st.block, workspace_lo, workspace_hi);
  // Residual overlap after both bodies move (not agent-only mid-step pen).
  pen = 0.f;
  deepest_contact(st.agent, st.block, st.ang, pen, n, contact, cog);
  return pen;
}

// Same signature spirit as pusht_physics::step_from_snapshot.
PUSHT_QS_HD void step_from_snapshot(const float *state_in, float ax, float ay,
                               float *state_out, int n_substeps = kDefaultSubsteps,
                               float *intermediates = nullptr,
                               float max_agent_vel = 0.f,
                               float *out_pen_increase = nullptr,
                               float support_radius = kDefaultSupportRadius,
                               float workspace_lo = 0.f,
                               float workspace_hi = 512.f,
                               float contact_mu = kContactFrictionMu) {
  State st = from_array(state_in);
  // Baseline geometric overlap (no dynamics).
  float pen0 = 0.f;
  {
    Vec2 n, ct, cog;
    deepest_contact(st.agent, st.block, st.ang, pen0, n, ct, cog);
  }
  float max_pen = pen0;
  for (int i = 0; i < n_substeps; ++i) {
    float pen = control_step(st, ax, ay, max_agent_vel, support_radius,
                             workspace_lo, workspace_hi, contact_mu);
    max_pen = qs_max(max_pen, pen);
    if (intermediates)
      to_array(st, intermediates + i * kStateDim);
  }
  to_array(st, state_out);
  if (out_pen_increase) {
    float inc = max_pen - pen0;
    *out_pen_increase = inc > 0.f ? inc : 0.f;
  }
}

// --- Host Succ action sampling (CPU env + CUDA fill_actions) -----------------

// Evenly spaced samples in [lo, hi], inclusive endpoints when n > 1.
inline float sample_interval(int i, int n, float lo, float hi) {
  if (n <= 1)
    return 0.5f * (lo + hi);
  return lo + (hi - lo) * (static_cast<float>(i) / static_cast<float>(n - 1));
}

inline void append_clamped_target(float ax, float ay, float r, float th,
                                  float ws_lo, float ws_hi,
                                  std::vector<float> &actions) {
  float tx = ax + r * qs_cos(th);
  float ty = ay + r * qs_sin(th);
  tx = qs_min(qs_max(tx, ws_lo), ws_hi);
  ty = qs_min(qs_max(ty, ws_lo), ws_hi);
  actions.push_back(tx);
  actions.push_back(ty);
}

// One agent-relative ring. μ = Chipmunk contact friction (fa*fb). If the agent
// overlaps the T, n_dir/2 angles fill the inward friction cone and the rest
// fill the outward leave-hemisphere; otherwise uniform on [0, 2π).
inline void append_qs_ring_targets(const float *pose, float r, int n_dir,
                                   float ws_lo, float ws_hi, float mu,
                                   std::vector<float> &actions) {
  const float ax = pose[0], ay = pose[1];
  float pen = 0.f;
  Vec2 n, contact, cog;
  const bool hit =
      deepest_contact({ax, ay}, {pose[2], pose[3]}, pose[4], pen, n, contact,
                      cog);
  if (!hit || n_dir <= 0) {
    for (int i = 0; i < n_dir; i++) {
      const float th =
          static_cast<float>(i) * (2.f * kPi / static_cast<float>(n_dir));
      append_clamped_target(ax, ay, r, th, ws_lo, ws_hi, actions);
    }
    return;
  }

  // n = outward from T (toward agent). Push into T along -n; leave along +n.
  const float ang_push = atan2f(-n.y, -n.x);
  const float ang_leave = atan2f(n.y, n.x);
  const float half_cone = atanf(qs_max(mu, 0.f)); // Coulomb limiting angle
  const int n_push = n_dir / 2;
  const int n_leave = n_dir - n_push;

  for (int i = 0; i < n_push; i++) {
    const float th =
        sample_interval(i, n_push, ang_push - half_cone, ang_push + half_cone);
    append_clamped_target(ax, ay, r, th, ws_lo, ws_hi, actions);
  }
  for (int i = 0; i < n_leave; i++) {
    const float th = sample_interval(i, n_leave, ang_leave - 0.5f * kPi,
                                     ang_leave + 0.5f * kPi);
    append_clamped_target(ax, ay, r, th, ws_lo, ws_hi, actions);
  }
}

} // namespace pusht_qs
