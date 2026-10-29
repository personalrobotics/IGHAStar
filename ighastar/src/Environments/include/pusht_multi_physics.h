// Chipmunk Push-T with two tees, an open right wall, and one static square.
// State [16]: ax ay | t0x t0y t0a | t1x t1y t1a | avx avy | t0vx t0vy t0w | t1vx t1vy t1w
// Clear bits are not in this vector. The Environment stores them on the node.
#pragma once

#include <chipmunk/chipmunk.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace pusht_multi_physics {

constexpr cpFloat kDt = 0.01;
constexpr cpFloat kP = 100.0;
constexpr cpFloat kV = 20.0;
constexpr cpFloat kAgentRadius = 15.0;
constexpr cpFloat kTeeScale = 30.0;
constexpr cpFloat kTeeLength = 4.0;
constexpr cpFloat kShapeFriction = 1.0;
constexpr int kNumTees = 2;
constexpr int kStateDim = 16;
constexpr int kDefaultSubsteps = 10;
constexpr cpCollisionType kTypeObs = 1;
constexpr cpCollisionType kTypeDynamic = 2;

inline float contact_friction_mu() {
  return static_cast<float>(kShapeFriction * kShapeFriction);
}

struct World {
  cpSpace *space = nullptr;
  cpBody *agent = nullptr;
  cpBody *tee[kNumTees] = {};
  cpShape *agent_shape = nullptr;
  cpShape *tee_shape[kNumTees][2] = {};
  cpShape *obstacle = nullptr;
  cpShape *wall[3] = {};
  cpFloat max_penetration = 0.0;
  int hit_obstacle = 0;
};

inline void pen_post_solve(cpArbiter *arb, cpSpace *, void *data) {
  auto *w = static_cast<World *>(data);
  const cpContactPointSet set = cpArbiterGetContactPointSet(arb);
  for (int i = 0; i < set.count; ++i) {
    if (set.points[i].distance < 0.0)
      w->max_penetration = std::max(w->max_penetration, -set.points[i].distance);
  }
}

inline cpBool obs_begin(cpArbiter *, cpSpace *, void *data) {
  static_cast<World *>(data)->hit_obstacle = 1;
  return cpTrue;
}

inline void install_handlers(World &w) {
  cpCollisionHandler *def = cpSpaceAddDefaultCollisionHandler(w.space);
  def->postSolveFunc = pen_post_solve;
  def->userData = &w;
  cpCollisionHandler *obs =
      cpSpaceAddCollisionHandler(w.space, kTypeObs, kTypeDynamic);
  obs->beginFunc = obs_begin;
  obs->postSolveFunc = pen_post_solve;
  obs->userData = &w;
}

inline cpShape *add_segment(cpSpace *space, cpVect a, cpVect b, cpFloat radius) {
  cpShape *shape = cpSegmentShapeNew(cpSpaceGetStaticBody(space), a, b, radius);
  cpShapeSetFriction(shape, kShapeFriction);
  cpSpaceAddShape(space, shape);
  return shape;
}

inline void add_tee(World &w, int index) {
  const cpFloat s = kTeeScale;
  const cpFloat L = kTeeLength;
  cpVect v1[4] = {cpv(-L * s / 2, s), cpv(L * s / 2, s), cpv(L * s / 2, 0),
                  cpv(-L * s / 2, 0)};
  cpVect v2[4] = {cpv(-s / 2, s), cpv(-s / 2, L * s), cpv(s / 2, L * s),
                  cpv(s / 2, s)};
  const cpFloat mass = 1.0;
  // Same inertia pair as single-T (gym-pusht uses v1 for both moments).
  const cpFloat i1 = cpMomentForPoly(mass, 4, v1, cpvzero, 0.0);
  const cpFloat i2 = cpMomentForPoly(mass, 4, v1, cpvzero, 0.0);
  cpBody *body = cpBodyNew(mass, i1 + i2);
  cpSpaceAddBody(w.space, body);
  cpShape *sh1 = cpPolyShapeNew(body, 4, v1, cpTransformIdentity, 0.0);
  cpShape *sh2 = cpPolyShapeNew(body, 4, v2, cpTransformIdentity, 0.0);
  cpShapeSetFriction(sh1, kShapeFriction);
  cpShapeSetFriction(sh2, kShapeFriction);
  cpShapeSetCollisionType(sh1, kTypeDynamic);
  cpShapeSetCollisionType(sh2, kTypeDynamic);
  cpSpaceAddShape(w.space, sh1);
  cpSpaceAddShape(w.space, sh2);
  const cpVect cog =
      cpvmult(cpvadd(cpCentroidForPoly(4, v1), cpCentroidForPoly(4, v2)), 0.5);
  cpBodySetCenterOfGravity(body, cog);
  w.tee[index] = body;
  w.tee_shape[index][0] = sh1;
  w.tee_shape[index][1] = sh2;
}

// Right wall (x = 506) is omitted. Tees leave through that gap.
inline World make_world(float obstacle_x, float obstacle_y, float obstacle_side) {
  World w;
  w.space = cpSpaceNew();
  cpSpaceSetGravity(w.space, cpvzero);
  cpSpaceSetDamping(w.space, 0.0);

  w.wall[0] = add_segment(w.space, cpv(5, 506), cpv(5, 5), 2);       // left
  w.wall[1] = add_segment(w.space, cpv(5, 5), cpv(506, 5), 2);       // bottom
  w.wall[2] = add_segment(w.space, cpv(5, 506), cpv(506, 506), 2);   // top

  w.agent = cpBodyNewKinematic();
  cpSpaceAddBody(w.space, w.agent);
  w.agent_shape = cpCircleShapeNew(w.agent, kAgentRadius, cpvzero);
  cpShapeSetFriction(w.agent_shape, kShapeFriction);
  cpShapeSetCollisionType(w.agent_shape, kTypeDynamic);
  cpSpaceAddShape(w.space, w.agent_shape);

  add_tee(w, 0);
  add_tee(w, 1);

  const cpFloat h = static_cast<cpFloat>(obstacle_side) * 0.5;
  cpVect sq[4] = {cpv(obstacle_x - h, obstacle_y - h),
                  cpv(obstacle_x + h, obstacle_y - h),
                  cpv(obstacle_x + h, obstacle_y + h),
                  cpv(obstacle_x - h, obstacle_y + h)};
  w.obstacle =
      cpPolyShapeNew(cpSpaceGetStaticBody(w.space), 4, sq, cpTransformIdentity, 0.0);
  cpShapeSetFriction(w.obstacle, kShapeFriction);
  cpShapeSetCollisionType(w.obstacle, kTypeObs);
  cpSpaceAddShape(w.space, w.obstacle);
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
  for (int i = 0; i < kNumTees; i++)
    cpBodyEachShape(w.tee[i], shape_free, w.space);
  cpBodyEachShape(cpSpaceGetStaticBody(w.space), shape_free, w.space);
  cpSpaceRemoveBody(w.space, w.agent);
  for (int i = 0; i < kNumTees; i++)
    cpSpaceRemoveBody(w.space, w.tee[i]);
  cpBodyFree(w.agent);
  for (int i = 0; i < kNumTees; i++)
    cpBodyFree(w.tee[i]);
  cpSpaceFree(w.space);
  w = {};
}

inline void remake_world(World &w, float ox, float oy, float side) {
  destroy_world(w);
  w = make_world(ox, oy, side);
}

inline void zero_accums(cpBody *body) {
  cpBodySetForce(body, cpvzero);
  cpBodySetTorque(body, 0.0);
}

inline int pose_base(int tee) { return 2 + tee * 3; }
inline int vel_base(int tee) { return 10 + tee * 3; }

inline void load_snapshot(World &w, const float *s, float ox, float oy,
                          float side) {
  remake_world(w, ox, oy, side);
  install_handlers(w);
  zero_accums(w.agent);
  cpBodySetVelocity(w.agent, cpv(s[8], s[9]));
  cpBodySetPosition(w.agent, cpv(s[0], s[1]));
  for (int i = 0; i < kNumTees; i++) {
    const int p = pose_base(i);
    const int v = vel_base(i);
    zero_accums(w.tee[i]);
    cpBodySetAngle(w.tee[i], s[p + 2]);
    cpBodySetPosition(w.tee[i], cpv(s[p], s[p + 1]));
    cpBodySetVelocity(w.tee[i], cpv(s[v], s[v + 1]));
    cpBodySetAngularVelocity(w.tee[i], s[v + 2]);
  }
  // cpShapesCollide reads cached shape transforms (circle->tc), not the body.
  cpSpaceReindexShapesForBody(w.space, w.agent);
  for (int i = 0; i < kNumTees; i++)
    cpSpaceReindexShapesForBody(w.space, w.tee[i]);
  w.hit_obstacle = 0;
  w.max_penetration = 0.0;
}

inline void get_state(const World &w, float *s) {
  const cpVect ap = cpBodyGetPosition(w.agent);
  const cpVect av = cpBodyGetVelocity(w.agent);
  s[0] = static_cast<float>(ap.x);
  s[1] = static_cast<float>(ap.y);
  s[8] = static_cast<float>(av.x);
  s[9] = static_cast<float>(av.y);
  for (int i = 0; i < kNumTees; i++) {
    const int p = pose_base(i);
    const int v = vel_base(i);
    const cpVect bp = cpBodyGetPosition(w.tee[i]);
    const cpVect bv = cpBodyGetVelocity(w.tee[i]);
    s[p] = static_cast<float>(bp.x);
    s[p + 1] = static_cast<float>(bp.y);
    cpFloat ang = std::fmod(cpBodyGetAngle(w.tee[i]), 2.0 * CP_PI);
    if (ang < 0)
      ang += 2.0 * CP_PI;
    s[p + 2] = static_cast<float>(ang);
    s[v] = static_cast<float>(bv.x);
    s[v + 1] = static_cast<float>(bv.y);
    s[v + 2] = static_cast<float>(cpBodyGetAngularVelocity(w.tee[i]));
  }
}

inline void clamp_agent_velocity(World &w, cpFloat max_speed) {
  if (max_speed <= 0)
    return;
  const cpVect vel = cpBodyGetVelocity(w.agent);
  const cpFloat speed = cpvlength(vel);
  if (speed > max_speed)
    cpBodySetVelocity(w.agent, cpvmult(vel, max_speed / speed));
}

inline void control_step(World &w, cpFloat ax, cpFloat ay, int n_substeps,
                         cpFloat max_agent_vel) {
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

inline bool shapes_overlap(cpShape *a, cpShape *b) {
  if (!a || !b)
    return false;
  const cpContactPointSet set = cpShapesCollide(a, b);
  for (int i = 0; i < set.count; i++) {
    if (set.points[i].distance < 0.0)
      return true;
  }
  return false;
}

// Start-state filter: overlapping shapes are invalid. Tee–tee contact during
// a push is allowed; that case is not checked here.
inline bool overlaps_forbidden(const float *s, float ox, float oy, float side) {
  thread_local World w{};
  load_snapshot(w, s, ox, oy, side);
  if (shapes_overlap(w.agent_shape, w.obstacle))
    return true;
  for (int i = 0; i < 3; i++) {
    if (shapes_overlap(w.agent_shape, w.wall[i]))
      return true;
  }
  for (int t = 0; t < kNumTees; t++) {
    for (int k = 0; k < 2; k++) {
      cpShape *sh = w.tee_shape[t][k];
      if (shapes_overlap(sh, w.obstacle))
        return true;
      if (shapes_overlap(sh, w.agent_shape))
        return true;
      for (int i = 0; i < 3; i++) {
        if (shapes_overlap(sh, w.wall[i]))
          return true;
      }
    }
  }
  for (int k = 0; k < 2; k++) {
    for (int m = 0; m < 2; m++) {
      if (shapes_overlap(w.tee_shape[0][k], w.tee_shape[1][m]))
        return true;
    }
  }
  return false;
}

inline void step_from_snapshot(const float *state_in, float ax, float ay,
                               float *state_out, int n_substeps, float *intermediates,
                               cpFloat max_agent_vel, float *out_pen_increase,
                               int *out_hit_obstacle, float ox, float oy,
                               float side) {
  thread_local World w{};
  load_snapshot(w, state_in, ox, oy, side);
  clamp_agent_velocity(w, max_agent_vel);

  w.max_penetration = 0.0;
  w.hit_obstacle = 0;
  cpSpaceStep(w.space, kDt);
  const cpFloat pen0 = w.max_penetration;

  load_snapshot(w, state_in, ox, oy, side);
  clamp_agent_velocity(w, max_agent_vel);
  w.max_penetration = 0.0;
  w.hit_obstacle = 0;
  for (int i = 0; i < n_substeps; ++i) {
    control_step(w, ax, ay, 1, max_agent_vel);
    if (intermediates)
      get_state(w, intermediates + i * kStateDim);
  }
  get_state(w, state_out);
  if (out_pen_increase) {
    const cpFloat inc = w.max_penetration - pen0;
    *out_pen_increase = static_cast<float>(inc > 0.0 ? inc : 0.0);
  }
  if (out_hit_obstacle)
    *out_hit_obstacle = w.hit_obstacle;
}

} // namespace pusht_multi_physics
