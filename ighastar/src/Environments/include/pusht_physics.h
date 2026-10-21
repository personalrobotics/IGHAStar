// gym-pusht / Diffusion Policy PushT physics (Chipmunk7), shared by the
// IGHAStar pusht Environment. See pusht_ext/README.md for legacy CoG notes.
#pragma once

#include <chipmunk/chipmunk.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace pusht_physics {

constexpr cpFloat kDt = 0.01;
constexpr cpFloat kP = 100.0;
constexpr cpFloat kV = 20.0;
constexpr cpFloat kAgentRadius = 15.0;
constexpr cpFloat kTeeScale = 30.0;
constexpr cpFloat kTeeLength = 4.0;
constexpr int kStateDim = 10; // ax ay bx by ang avx avy bvx bvy bw
constexpr int kDefaultSubsteps = 10;

struct World {
  cpSpace *space = nullptr;
  cpBody *agent = nullptr;
  cpBody *block = nullptr;
  // Max -distance over contact points this edge (pixels). Reset per step_from_snapshot.
  cpFloat max_penetration = 0.0;
};

// Track deepest overlap (distance < 0 → penetration).
inline void pen_post_solve(cpArbiter *arb, cpSpace *, void *data) {
  auto *max_pen = static_cast<cpFloat *>(data);
  const cpContactPointSet set = cpArbiterGetContactPointSet(arb);
  for (int i = 0; i < set.count; ++i) {
    if (set.points[i].distance < 0.0)
      *max_pen = std::max(*max_pen, -set.points[i].distance);
  }
}

inline void install_pen_handler(World &w) {
  cpCollisionHandler *h = cpSpaceAddDefaultCollisionHandler(w.space);
  h->postSolveFunc = pen_post_solve;
  h->userData = &w.max_penetration;
}

inline void add_segment(cpSpace *space, cpVect a, cpVect b, cpFloat radius) {
  cpShape *shape = cpSegmentShapeNew(cpSpaceGetStaticBody(space), a, b, radius);
  cpShapeSetFriction(shape, 1.0);
  cpSpaceAddShape(space, shape);
}

inline World make_world() {
  World w;
  w.space = cpSpaceNew();
  cpSpaceSetGravity(w.space, cpvzero);
  cpSpaceSetDamping(w.space, 0.0);

  add_segment(w.space, cpv(5, 506), cpv(5, 5), 2);
  add_segment(w.space, cpv(5, 5), cpv(506, 5), 2);
  add_segment(w.space, cpv(506, 5), cpv(506, 506), 2);
  add_segment(w.space, cpv(5, 506), cpv(506, 506), 2);

  w.agent = cpBodyNewKinematic();
  cpSpaceAddBody(w.space, w.agent);
  cpShape *agent_shape = cpCircleShapeNew(w.agent, kAgentRadius, cpvzero);
  cpShapeSetFriction(agent_shape, 1.0);
  cpSpaceAddShape(w.space, agent_shape);

  const cpFloat s = kTeeScale;
  const cpFloat L = kTeeLength;
  cpVect v1[4] = {cpv(-L * s / 2, s), cpv(L * s / 2, s), cpv(L * s / 2, 0),
                  cpv(-L * s / 2, 0)};
  cpVect v2[4] = {cpv(-s / 2, s), cpv(-s / 2, L * s), cpv(s / 2, L * s),
                  cpv(s / 2, s)};
  const cpFloat mass = 1.0;
  // LEGACY: inertia2 uses v1 (gym-pusht bug)
  const cpFloat i1 = cpMomentForPoly(mass, 4, v1, cpvzero, 0.0);
  const cpFloat i2 = cpMomentForPoly(mass, 4, v1, cpvzero, 0.0);
  w.block = cpBodyNew(mass, i1 + i2);
  cpSpaceAddBody(w.space, w.block);

  cpShape *sh1 = cpPolyShapeNew(w.block, 4, v1, cpTransformIdentity, 0.0);
  cpShape *sh2 = cpPolyShapeNew(w.block, 4, v2, cpTransformIdentity, 0.0);
  cpShapeSetFriction(sh1, 1.0);
  cpShapeSetFriction(sh2, 1.0);
  cpSpaceAddShape(w.space, sh1);
  cpSpaceAddShape(w.space, sh2);

  const cpVect cog =
      cpvmult(cpvadd(cpCentroidForPoly(4, v1), cpCentroidForPoly(4, v2)), 0.5);
  cpBodySetCenterOfGravity(w.block, cog);
  install_pen_handler(w);
  return w;
}

inline void shape_free(cpBody *, cpShape *shape, void *data) {
  auto *space = static_cast<cpSpace *>(data);
  cpSpaceRemoveShape(space, shape);
  cpShapeFree(shape);
}

inline void destroy_world(World &w) {
  if (!w.space)
    return;
  cpBodyEachShape(w.agent, shape_free, w.space);
  cpBodyEachShape(w.block, shape_free, w.space);
  cpBodyEachShape(cpSpaceGetStaticBody(w.space), shape_free, w.space);
  cpSpaceRemoveBody(w.space, w.agent);
  cpSpaceRemoveBody(w.space, w.block);
  cpBodyFree(w.agent);
  cpBodyFree(w.block);
  cpSpaceFree(w.space);
  w = {};
}

inline void remake_world(World &w) {
  destroy_world(w);
  w = make_world();
}

inline void zero_accums(cpBody *body) {
  cpBodySetForce(body, cpvzero);
  cpBodySetTorque(body, 0.0);
}

// Snapshot restore: angle THEN position (planner / get_obs values).
inline void load_snapshot(World &w, const float *s) {
  remake_world(w);
  zero_accums(w.agent);
  zero_accums(w.block);
  cpBodySetVelocity(w.agent, cpv(s[5], s[6]));
  cpBodySetAngle(w.block, s[4]);
  cpBodySetPosition(w.block, cpv(s[2], s[3]));
  cpBodySetVelocity(w.block, cpv(s[7], s[8]));
  cpBodySetAngularVelocity(w.block, s[9]);
  cpBodySetPosition(w.agent, cpv(s[0], s[1]));
}

inline void get_state(const World &w, float *s) {
  const cpVect ap = cpBodyGetPosition(w.agent);
  const cpVect av = cpBodyGetVelocity(w.agent);
  const cpVect bp = cpBodyGetPosition(w.block);
  const cpVect bv = cpBodyGetVelocity(w.block);
  s[0] = static_cast<float>(ap.x);
  s[1] = static_cast<float>(ap.y);
  s[2] = static_cast<float>(bp.x);
  s[3] = static_cast<float>(bp.y);
  cpFloat ang = std::fmod(cpBodyGetAngle(w.block), 2.0 * CP_PI);
  if (ang < 0)
    ang += 2.0 * CP_PI;
  s[4] = static_cast<float>(ang);
  s[5] = static_cast<float>(av.x);
  s[6] = static_cast<float>(av.y);
  s[7] = static_cast<float>(bv.x);
  s[8] = static_cast<float>(bv.y);
  s[9] = static_cast<float>(cpBodyGetAngularVelocity(w.block));
}

// Clamp agent speed. max_speed <= 0 disables (legacy gym-pusht / dino_wm).
inline void clamp_agent_velocity(World &w, cpFloat max_speed) {
  if (max_speed <= 0)
    return;
  const cpVect vel = cpBodyGetVelocity(w.agent);
  const cpFloat speed = cpvlength(vel);
  if (speed > max_speed)
    cpBodySetVelocity(w.agent, cpvmult(vel, max_speed / speed));
}

inline void control_step(World &w, cpFloat ax, cpFloat ay, int n_substeps,
                         cpFloat max_agent_vel = 0.0) {
  for (int i = 0; i < n_substeps; ++i) {
    const cpVect pos = cpBodyGetPosition(w.agent);
    const cpVect vel = cpBodyGetVelocity(w.agent);
    const cpVect acc = cpv(kP * (ax - pos.x) + kV * (0.0 - vel.x),
                           kP * (ay - pos.y) + kV * (0.0 - vel.y));
    cpBodySetVelocity(w.agent, cpv(vel.x + acc.x * kDt, vel.y + acc.y * kDt));
    clamp_agent_velocity(w, max_agent_vel);
    cpSpaceStep(w.space, kDt);
  }
}

// One gym env.step from a snapshot (independent edge).
// If intermediates != nullptr, writes n_substeps states (after each dt) there.
// If out_pen_increase != nullptr, writes how much deeper contacts got vs a
// one-step baseline after load (px). Soft contacts often already overlap a few
// px at rest; we only reject edges that *deepen* penetration.
inline void step_from_snapshot(const float *state_in, float ax, float ay,
                               float *state_out, int n_substeps = kDefaultSubsteps,
                               float *intermediates = nullptr,
                               cpFloat max_agent_vel = 0.0,
                               float *out_pen_increase = nullptr) {
  thread_local World w{};
  load_snapshot(w, state_in);
  // Re-bind handler userData after remake (World address is stable: thread_local).
  install_pen_handler(w);
  clamp_agent_velocity(w, max_agent_vel);

  // Baseline overlap at the start of the edge (Chipmunk soft contacts).
  // One step then reload so the real edge still begins from state_in.
  w.max_penetration = 0.0;
  cpSpaceStep(w.space, kDt);
  const cpFloat pen0 = w.max_penetration;

  load_snapshot(w, state_in);
  install_pen_handler(w);
  clamp_agent_velocity(w, max_agent_vel);
  w.max_penetration = 0.0;
  for (int i = 0; i < n_substeps; ++i) {
    control_step(w, ax, ay, /*n_substeps=*/1, max_agent_vel);
    if (intermediates) {
      get_state(w, intermediates + i * kStateDim);
    }
  }
  get_state(w, state_out);
  if (out_pen_increase) {
    const cpFloat inc = w.max_penetration - pen0;
    *out_pen_increase = static_cast<float>(inc > 0.0 ? inc : 0.0);
  }
}

} // namespace pusht_physics
