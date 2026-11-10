// Chipmunk Push-T with two tees, K agents, and interior wall segments.
// State: agent xy (K) | tee poses | agent vel (K) | tee vel.
// K=1 is ax ay | t0 t1 | avx avy | t0v t1v. K comes from PUSHT_MULTI_AGENTS.
// Clear bits are not in this vector. The Environment stores them on the node.
// A wall is one segment (x1, y1, x2, y2) in the same pixels as the tee.
// The box around the workspace is separate and is not one of these segments.
// Contact with a wall is allowed. The Environment rejects an edge when the
// contact force on the agent exceeds its threshold.
#pragma once

#include <chipmunk/chipmunk.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifndef PUSHT_MULTI_AGENTS
#define PUSHT_MULTI_AGENTS 1
#endif

// The agent count is a compile flag. Two extensions (1 agent and 2) can be
// loaded in one process, so the symbols must not collide.
#define PUSHT_MULTI_NS_CAT(a, b) pusht_multi_##a##_##b
#define PUSHT_MULTI_NS(k) PUSHT_MULTI_NS_CAT(physics, k)
namespace PUSHT_MULTI_NS(PUSHT_MULTI_AGENTS) {

constexpr cpFloat kDt = 0.01;
constexpr cpFloat kP = 100.0;
constexpr cpFloat kV = 20.0;
constexpr cpFloat kAgentRadius = 15.0;
constexpr cpFloat kTeeScale = 30.0;
constexpr cpFloat kTeeLength = 4.0;
constexpr cpFloat kShapeFriction = 1.0;
constexpr int kNumAgents = PUSHT_MULTI_AGENTS;
constexpr int kNumTees = 2;
constexpr int kStateDim = 4 * kNumAgents + 6 * kNumTees;
constexpr int kDefaultSubsteps = 10;
constexpr cpCollisionType kTypeDynamic = 2;
constexpr int kMaxWalls = 32;
constexpr int kMaxCutSegs = kMaxWalls;
constexpr float kBoxLo = 5.f;
constexpr float kBoxHi = 506.f;

// One interior segment. Endpoints are pixels, same frame as the tee pose.
struct WallSpec {
  float x1;
  float y1;
  float x2;
  float y2;
};

inline float contact_friction_mu() {
  return static_cast<float>(kShapeFriction * kShapeFriction);
}

inline int agent_pose(int i) { return 2 * i; }
inline int tee_pose(int i) { return 2 * kNumAgents + 3 * i; }
inline int agent_vel(int i) { return 2 * kNumAgents + 3 * kNumTees + 2 * i; }
inline int tee_vel(int i) {
  return 2 * kNumAgents + 3 * kNumTees + 2 * kNumAgents + 3 * i;
}

struct World {
  cpSpace *space = nullptr;
  cpBody *agent[kNumAgents] = {};
  cpBody *tee[kNumTees] = {};
  cpShape *agent_shape[kNumAgents] = {};
  cpShape *tee_shape[kNumTees][2] = {};
  cpShape *wall[4] = {};
  int n_walls = 0;
  cpShape *cut[kMaxCutSegs] = {};
  int n_cut = 0;
  cpFloat max_penetration = 0.0;
  cpFloat max_agent_force = 0.0; // peak |contact impulse| / dt on the agent
};

inline void pen_post_solve(cpArbiter *arb, cpSpace *, void *data) {
  auto *w = static_cast<World *>(data);
  const cpContactPointSet set = cpArbiterGetContactPointSet(arb);
  for (int i = 0; i < set.count; ++i) {
    if (set.points[i].distance < 0.0)
      w->max_penetration = std::max(w->max_penetration, -set.points[i].distance);
  }
  cpBody *ba = nullptr;
  cpBody *bb = nullptr;
  cpArbiterGetBodies(arb, &ba, &bb);
  bool on_agent = false;
  for (int i = 0; i < kNumAgents; i++)
    on_agent = on_agent || ba == w->agent[i] || bb == w->agent[i];
  if (on_agent) {
    const cpVect impulse = cpArbiterTotalImpulse(arb);
    const cpFloat force = cpvlength(impulse) / kDt;
    if (force > w->max_agent_force)
      w->max_agent_force = force;
  }
}

inline void install_handlers(World &w) {
  cpCollisionHandler *def = cpSpaceAddDefaultCollisionHandler(w.space);
  def->postSolveFunc = pen_post_solve;
  def->userData = &w;
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

inline void add_cut_piece(World &w, float x1, float y1, float x2, float y2) {
  if (std::hypot(x2 - x1, y2 - y1) < 1.0)
    return;
  if (w.n_cut >= kMaxCutSegs)
    throw std::runtime_error("pusht_multi: too many wall segments");
  w.cut[w.n_cut++] = add_segment(w.space, cpv(x1, y1), cpv(x2, y2), 2);
}

inline void add_cut_wall(World &w, WallSpec spec) {
  add_cut_piece(w, spec.x1, spec.y1, spec.x2, spec.y2);
}

// include_right_wall false: gap at x=506 (clear). true: closed box (rearrange).
inline World make_world(const WallSpec *walls, int n_walls,
                        bool include_right_wall = false) {
  World w;
  w.space = cpSpaceNew();
  cpSpaceSetGravity(w.space, cpvzero);
  cpSpaceSetDamping(w.space, 0.0);

  w.wall[0] = add_segment(w.space, cpv(kBoxLo, kBoxHi), cpv(kBoxLo, kBoxLo), 2);
  w.wall[1] = add_segment(w.space, cpv(kBoxLo, kBoxLo), cpv(kBoxHi, kBoxLo), 2);
  w.wall[2] = add_segment(w.space, cpv(kBoxLo, kBoxHi), cpv(kBoxHi, kBoxHi), 2);
  w.n_walls = 3;
  if (include_right_wall) {
    w.wall[3] = add_segment(w.space, cpv(kBoxHi, kBoxLo), cpv(kBoxHi, kBoxHi), 2);
    w.n_walls = 4;
  }
  if (n_walls > kMaxWalls)
    throw std::runtime_error("pusht_multi: too many wall segments");
  for (int i = 0; i < n_walls; i++)
    add_cut_wall(w, walls[i]);

  for (int i = 0; i < kNumAgents; i++) {
    w.agent[i] = cpBodyNewKinematic();
    cpSpaceAddBody(w.space, w.agent[i]);
    w.agent_shape[i] = cpCircleShapeNew(w.agent[i], kAgentRadius, cpvzero);
    cpShapeSetFriction(w.agent_shape[i], kShapeFriction);
    cpShapeSetCollisionType(w.agent_shape[i], kTypeDynamic);
    cpSpaceAddShape(w.space, w.agent_shape[i]);
  }

  add_tee(w, 0);
  add_tee(w, 1);

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
  for (int i = 0; i < kNumAgents; i++)
    cpBodyEachShape(w.agent[i], shape_free, w.space);
  for (int i = 0; i < kNumTees; i++)
    cpBodyEachShape(w.tee[i], shape_free, w.space);
  cpBodyEachShape(cpSpaceGetStaticBody(w.space), shape_free, w.space);
  for (int i = 0; i < kNumAgents; i++)
    cpSpaceRemoveBody(w.space, w.agent[i]);
  for (int i = 0; i < kNumTees; i++)
    cpSpaceRemoveBody(w.space, w.tee[i]);
  for (int i = 0; i < kNumAgents; i++)
    cpBodyFree(w.agent[i]);
  for (int i = 0; i < kNumTees; i++)
    cpBodyFree(w.tee[i]);
  cpSpaceFree(w.space);
  w = {};
}

inline void remake_world(World &w, const WallSpec *walls, int n_walls,
                         bool include_right_wall = false) {
  destroy_world(w);
  w = make_world(walls, n_walls, include_right_wall);
}

inline void zero_accums(cpBody *body) {
  cpBodySetForce(body, cpvzero);
  cpBodySetTorque(body, 0.0);
}

inline void load_snapshot(World &w, const float *s, const WallSpec *walls,
                          int n_walls, bool include_right_wall = false) {
  remake_world(w, walls, n_walls, include_right_wall);
  install_handlers(w);
  for (int i = 0; i < kNumAgents; i++) {
    const int p = agent_pose(i);
    const int v = agent_vel(i);
    zero_accums(w.agent[i]);
    cpBodySetVelocity(w.agent[i], cpv(s[v], s[v + 1]));
    cpBodySetPosition(w.agent[i], cpv(s[p], s[p + 1]));
  }
  for (int i = 0; i < kNumTees; i++) {
    const int p = tee_pose(i);
    const int v = tee_vel(i);
    zero_accums(w.tee[i]);
    cpBodySetAngle(w.tee[i], s[p + 2]);
    cpBodySetPosition(w.tee[i], cpv(s[p], s[p + 1]));
    cpBodySetVelocity(w.tee[i], cpv(s[v], s[v + 1]));
    cpBodySetAngularVelocity(w.tee[i], s[v + 2]);
  }
  // cpShapesCollide reads cached shape transforms (circle->tc), not the body.
  for (int i = 0; i < kNumAgents; i++)
    cpSpaceReindexShapesForBody(w.space, w.agent[i]);
  for (int i = 0; i < kNumTees; i++)
    cpSpaceReindexShapesForBody(w.space, w.tee[i]);
  w.max_penetration = 0.0;
  w.max_agent_force = 0.0;
}

inline void get_state(const World &w, float *s) {
  for (int i = 0; i < kNumAgents; i++) {
    const int p = agent_pose(i);
    const int v = agent_vel(i);
    const cpVect ap = cpBodyGetPosition(w.agent[i]);
    const cpVect av = cpBodyGetVelocity(w.agent[i]);
    s[p] = static_cast<float>(ap.x);
    s[p + 1] = static_cast<float>(ap.y);
    s[v] = static_cast<float>(av.x);
    s[v + 1] = static_cast<float>(av.y);
  }
  for (int i = 0; i < kNumTees; i++) {
    const int p = tee_pose(i);
    const int v = tee_vel(i);
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

inline void clamp_body_speed(cpBody *body, cpFloat max_speed) {
  if (max_speed <= 0)
    return;
  const cpVect vel = cpBodyGetVelocity(body);
  const cpFloat speed = cpvlength(vel);
  if (speed > max_speed)
    cpBodySetVelocity(body, cpvmult(vel, max_speed / speed));
}

// The agent is kinematic, so a collision does not move it. Put the circle back
// on the near side of any wall it entered this step.
inline void push_circle_out(cpBody *body, cpFloat radius, cpShape *seg) {
  if (!seg)
    return;
  const cpVect a = cpSegmentShapeGetA(seg);
  const cpVect b = cpSegmentShapeGetB(seg);
  const cpFloat min_dist = radius + cpSegmentShapeGetRadius(seg);
  const cpVect c = cpBodyGetPosition(body);
  const cpVect ab = cpvsub(b, a);
  const cpFloat ab_len2 = cpvdot(ab, ab);
  cpFloat t = 0.0;
  if (ab_len2 > 1e-8)
    t = cpvdot(cpvsub(c, a), ab) / ab_len2;
  if (t < 0.0)
    t = 0.0;
  if (t > 1.0)
    t = 1.0;
  const cpVect closest = cpvadd(a, cpvmult(ab, t));
  const cpVect delta = cpvsub(c, closest);
  const cpFloat dist = cpvlength(delta);
  if (dist >= min_dist)
    return;
  cpVect n;
  if (dist > 1e-6) {
    n = cpvmult(delta, 1.0 / dist);
  } else {
    const cpVect vel = cpBodyGetVelocity(body);
    const cpFloat speed = cpvlength(vel);
    n = speed > 1e-6 ? cpvmult(vel, -1.0 / speed) : cpv(1.0, 0.0);
  }
  cpBodySetPosition(body, cpvadd(closest, cpvmult(n, min_dist)));
}

// Both agents are kinematic, so a collision does not separate them.
inline void separate_agents(World &w) {
  const cpFloat min_dist = 2.0 * kAgentRadius;
  for (int i = 0; i < kNumAgents; i++) {
    for (int j = i + 1; j < kNumAgents; j++) {
      const cpVect pi = cpBodyGetPosition(w.agent[i]);
      const cpVect pj = cpBodyGetPosition(w.agent[j]);
      const cpVect d = cpvsub(pi, pj);
      const cpFloat dist = cpvlength(d);
      if (dist >= min_dist)
        continue;
      const cpVect n = dist > 1e-6 ? cpvmult(d, 1.0 / dist) : cpv(1.0, 0.0);
      const cpFloat push = 0.5 * (min_dist - dist);
      cpBodySetPosition(w.agent[i], cpvadd(pi, cpvmult(n, push)));
      cpBodySetPosition(w.agent[j], cpvadd(pj, cpvmult(n, -push)));
    }
  }
}

inline void keep_agents_out_of_walls(World &w) {
  for (int pass = 0; pass < 2; ++pass) {
    for (int a = 0; a < kNumAgents; a++) {
      for (int i = 0; i < w.n_walls; i++)
        push_circle_out(w.agent[a], kAgentRadius, w.wall[i]);
      for (int i = 0; i < w.n_cut; i++)
        push_circle_out(w.agent[a], kAgentRadius, w.cut[i]);
    }
    separate_agents(w);
  }
  for (int a = 0; a < kNumAgents; a++)
    cpSpaceReindexShapesForBody(w.space, w.agent[a]);
}

// target is K pairs of xy. An agent whose target is its current position holds.
inline void control_step(World &w, const float *target, int n_substeps,
                         cpFloat max_agent_vel) {
  for (int step = 0; step < n_substeps; ++step) {
    for (int i = 0; i < kNumAgents; i++) {
      const cpVect pos = cpBodyGetPosition(w.agent[i]);
      const cpVect vel = cpBodyGetVelocity(w.agent[i]);
      const cpFloat ax = target[2 * i];
      const cpFloat ay = target[2 * i + 1];
      const cpVect acc = cpv(kP * (ax - pos.x) + kV * (0.0 - vel.x),
                             kP * (ay - pos.y) + kV * (0.0 - vel.y));
      cpBodySetVelocity(w.agent[i], cpv(vel.x + acc.x * kDt, vel.y + acc.y * kDt));
      clamp_body_speed(w.agent[i], max_agent_vel);
    }
    cpSpaceStep(w.space, kDt);
    keep_agents_out_of_walls(w);
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

// cpShapesCollide misses a segment that enters one side of a polygon and
// leaves the other. The segment's bounds are correct, so test those.
inline bool segment_hits_bb(cpBB bb, cpVect a, cpVect b, cpFloat radius) {
  bb.l -= radius;
  bb.r += radius;
  bb.b -= radius;
  bb.t += radius;
  const float x0 = static_cast<float>(std::min(a.x, b.x));
  const float x1 = static_cast<float>(std::max(a.x, b.x));
  const float y0 = static_cast<float>(std::min(a.y, b.y));
  const float y1 = static_cast<float>(std::max(a.y, b.y));
  return x1 >= bb.l && x0 <= bb.r && y1 >= bb.b && y0 <= bb.t;
}

inline bool shape_hits_segment(cpShape *shape, cpShape *segment) {
  if (!shape || !segment)
    return false;
  return segment_hits_bb(cpShapeGetBB(shape), cpSegmentShapeGetA(segment),
                         cpSegmentShapeGetB(segment),
                         cpSegmentShapeGetRadius(segment));
}

// Start-state filter: overlapping shapes are invalid. Contact during a push
// is allowed. Tee–tee overlap at the start is not.
inline bool overlaps_forbidden(const float *s, const WallSpec *walls, int n_walls,
                               bool include_right_wall = false) {
  thread_local World w{};
  load_snapshot(w, s, walls, n_walls, include_right_wall);
  for (int a = 0; a < kNumAgents; a++) {
    for (int i = 0; i < w.n_walls; i++) {
      if (shapes_overlap(w.agent_shape[a], w.wall[i]))
        return true;
    }
    for (int i = 0; i < w.n_cut; i++) {
      if (shapes_overlap(w.agent_shape[a], w.cut[i]))
        return true;
    }
    for (int b = a + 1; b < kNumAgents; b++) {
      if (shapes_overlap(w.agent_shape[a], w.agent_shape[b]))
        return true;
    }
  }
  for (int t = 0; t < kNumTees; t++) {
    for (int k = 0; k < 2; k++) {
      cpShape *sh = w.tee_shape[t][k];
      for (int a = 0; a < kNumAgents; a++) {
        if (shapes_overlap(sh, w.agent_shape[a]))
          return true;
      }
      for (int i = 0; i < w.n_walls; i++) {
        if (shape_hits_segment(sh, w.wall[i]))
          return true;
      }
      for (int i = 0; i < w.n_cut; i++) {
        if (shape_hits_segment(sh, w.cut[i]))
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

inline void step_from_snapshot(const float *state_in, const float *target,
                               float *state_out, int n_substeps, float *intermediates,
                               cpFloat max_agent_vel, float *out_pen_increase,
                               float *out_peak_force, const WallSpec *walls,
                               int n_walls, bool include_right_wall = false) {
  thread_local World w{};
  load_snapshot(w, state_in, walls, n_walls, include_right_wall);
  for (int i = 0; i < kNumAgents; i++)
    clamp_body_speed(w.agent[i], max_agent_vel);

  w.max_penetration = 0.0;
  w.max_agent_force = 0.0;
  cpSpaceStep(w.space, kDt);
  const cpFloat pen0 = w.max_penetration;

  load_snapshot(w, state_in, walls, n_walls, include_right_wall);
  for (int i = 0; i < kNumAgents; i++)
    clamp_body_speed(w.agent[i], max_agent_vel);
  w.max_penetration = 0.0;
  w.max_agent_force = 0.0;
  for (int i = 0; i < n_substeps; ++i) {
    control_step(w, target, 1, max_agent_vel);
    if (intermediates)
      get_state(w, intermediates + i * kStateDim);
  }
  get_state(w, state_out);
  if (out_pen_increase) {
    const cpFloat inc = w.max_penetration - pen0;
    *out_pen_increase = static_cast<float>(inc > 0.0 ? inc : 0.0);
  }
  if (out_peak_force)
    *out_peak_force = static_cast<float>(w.max_agent_force);
}

} // namespace pusht_multi_physics_K
namespace pusht_multi_physics = PUSHT_MULTI_NS(PUSHT_MULTI_AGENTS);
