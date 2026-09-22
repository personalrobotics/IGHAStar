#include "terrain_map.h"
#include <algorithm>
#include <cfloat>
#include <cmath>

#define x_index 0
#define y_index 1
#define yaw_index 2
#define vx_index 3
#define vy_index 4
#define omega_index 5

#define ax_index 0
#define ay_index 1
#define alpha_index 2
#define GRAVITY 9.81f

__device__ float nan_to_num(float x, float replace) {
  return (std::isnan(x) || std::isinf(x)) ? replace : x;
}

__device__ float clamp(float x, float lower, float upper) {
  return std::min(std::max(x, lower), upper);
}

__device__ float wrap_to_pi(float x) {
  return std::fmod(x + M_PI, 2 * M_PI) - M_PI;
}

#define MAX_MARCH_STEPS 32

__device__ int esdf_column(float x, float y, const TerrainMap &map) {
  int img_X = clamp(static_cast<int>((x * map.res_inv)), 0, map.nx - 1);
  int img_Y = clamp(static_cast<int>((y * map.res_inv)), 0, map.ny - 1);
  return (img_Y * map.nx + img_X) * map.nz;
}

__device__ float esdf_surface_z(float x, float y, const TerrainMap &map) {
  const int column = esdf_column(x, y, map);
  int k = map.nz - 1;
  float d = map.esdf[column + k];
  for (int step = 0; step < MAX_MARCH_STEPS && d > 0.0f && k > 0; step++) {
    k = max(k - max(1, static_cast<int>(d / map.voxel_z)), 0);
    d = map.esdf[column + k];
  }
  float d_lo = d;
  float d_hi = map.esdf[column + min(k + 1, map.nz - 1)];
  float delta = d_hi - d_lo;
  float frac = (fabsf(delta) > 1e-12f) ? (-d_lo / delta) : 0.0f;
  frac = clamp(frac, 0.0f, 1.0f);
  return map.z_min + (float(k) + frac) * map.voxel_z;
}

__device__ float map_to_elev(float x, float y, const TerrainMap &map) {
  if (map.type == MAP_ELEVATION) {
    int img_X = clamp(static_cast<int>((x * map.res_inv)), 0, map.nx - 1);
    int img_Y = clamp(static_cast<int>((y * map.res_inv)), 0, map.ny - 1);
    return map.elev[img_Y * map.nx + img_X];
  }
  return esdf_surface_z(x, y, map);
}

__device__ float map_to_cost(float x, float y, const TerrainMap &map) {
  if (map.type == MAP_ELEVATION) {
    int img_X = clamp(static_cast<int>((x * map.res_inv)), 0, map.nx - 1);
    int img_Y = clamp(static_cast<int>((y * map.res_inv)), 0, map.ny - 1);
    return map.cost[img_Y * map.nx + img_X];
  }
  const int column = esdf_column(x, y, map);
  int k = clamp(static_cast<int>((esdf_surface_z(x, y, map) - map.z_min) /
                                 map.voxel_z),
                0, map.nz - 1);
  uchar4 c = map.color[column + k];
  return 0.299f * float(c.x) + 0.587f * float(c.y) + 0.114f * float(c.z);
}

__device__ void get_footprint_z(float *fl, float *fr, float *bl, float *br,
                                float &z, float x, float y, float cy, float sy,
                                const TerrainMap &map, float car_l2,
                                float car_w2) {
  fl[0] = car_l2 * cy - car_w2 * sy + x;
  fl[1] = car_l2 * sy + car_w2 * cy + y;

  fr[0] = car_l2 * cy - (-1) * car_w2 * sy + x;
  fr[1] = car_l2 * sy + (-1) * car_w2 * cy + y;

  bl[0] = (-1) * car_l2 * cy - car_w2 * sy + x;
  bl[1] = (-1) * car_l2 * sy + car_w2 * cy + y;

  br[0] = (-1) * car_l2 * cy - (-1) * car_w2 * sy + x;
  br[1] = (-1) * car_l2 * sy + (-1) * car_w2 * cy + y;

  float z_cent = map_to_elev(0, 0, map);
  z = map_to_elev(x, y, map) - z_cent;

  fl[2] = map_to_elev(fl[0], fl[1], map) - z_cent;
  fr[2] = map_to_elev(fr[0], fr[1], map) - z_cent;
  bl[2] = map_to_elev(bl[0], bl[1], map) - z_cent;
  br[2] = map_to_elev(br[0], br[1], map) - z_cent;
}

__global__ void
check_validity_batch_kernel(TerrainMap map, float *d_intermediate_states,
                            int patch_length_px, int patch_width_px,
                            float car_l2, float car_w2, int NX, int timesteps,
                            bool *d_valid) {
  int t = blockIdx.x;
  int k = blockIdx.y;
  int i = threadIdx.x;
  int j = threadIdx.y;

  if (i >= patch_length_px || j >= patch_width_px)
    return;
  if (!(d_valid[k]))
    return;

  int intermediate_index = k * timesteps * NX + t * NX;

  float x = d_intermediate_states[intermediate_index + x_index];
  float y = d_intermediate_states[intermediate_index + y_index];
  float theta = d_intermediate_states[intermediate_index + yaw_index];

  float cy = cosf(theta);
  float sy = sinf(theta);
  float offset_x = (i * map.voxel_xy) - car_l2;
  float offset_y = (j * map.voxel_xy) - car_w2;

  float px = offset_x * cy - offset_y * sy + x;
  float py = offset_x * sy + offset_y * cy + y;

  if (px < 0 || px >= map.nx * map.voxel_xy || py < 0 ||
      py >= map.ny * map.voxel_xy) {
    d_valid[k] = false;
    return;
  }

  if (map_to_cost(px, py, map) <= 250.0f) {
    d_valid[k] = false;
  }
}

__global__ void holonomic_6d_kernel(float *state, float *intermediate_states,
                                    float *controls, TerrainMap map, bool *valid,
                                    float *cost, float dt, int timesteps,
                                    int rollouts, int NX, int NC, float car_l2,
                                    float car_w2, float accel_max,
                                    float alpha_max, float max_vel, float min_vel,
                                    float max_omega, float RI,
                                    float max_vert_acc, float max_theta) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (k >= rollouts)
    return;

  int state_base = k * NX;
  int intermediate_index;

  float x = state[state_base + x_index];
  float y = state[state_base + y_index];
  float yaw = state[state_base + yaw_index];
  float vx_body = state[state_base + vx_index];
  float vy_body = state[state_base + vy_index];
  float omega = state[state_base + omega_index];

  float ax_body, ay_body, alpha;

  float cy = cosf(yaw), sy = sinf(yaw);
  float fl[3], fr[3], bl[3], br[3], z;
  get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, map, car_l2, car_w2);
  float last_roll = atan2f((fl[2] + bl[2]) - (fr[2] + br[2]), 4 * car_w2);
  float last_pitch = atan2f((bl[2] + br[2]) - (fl[2] + fr[2]), 4 * car_l2);
  float roll, pitch, wx, wy, cp, sp, cr, sr, ct;
  float vx_world, vy_world, ay_gravity, az;

  valid[k] = true;

  for (int t = 1; t <= timesteps; t++) {
    int control_base = k * timesteps * NC + (t - 1) * NC;
    ax_body = controls[control_base + ax_index] * accel_max;
    ay_body = controls[control_base + ay_index] * accel_max;
    alpha = controls[control_base + alpha_index] * alpha_max;

    cy = cosf(yaw);
    sy = sinf(yaw);
    get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, map, car_l2, car_w2);
    roll = atan2f((fl[2] + bl[2]) - (fr[2] + br[2]), 4 * car_w2);
    pitch = atan2f((bl[2] + br[2]) - (fl[2] + fr[2]), 4 * car_l2);

    wx = (roll - last_roll) / dt;
    wy = (pitch - last_pitch) / dt;
    last_pitch = pitch;
    last_roll = roll;

    cp = cosf(pitch);
    sp = sinf(pitch);
    cr = cosf(roll);
    sr = sinf(roll);
    ct = nan_to_num(sqrtf(1.0f - sp * sp - sr * sr), 0.0f);

    vx_body += (ax_body + sp * GRAVITY) * dt;
    vy_body += (ay_body - sr * GRAVITY) * dt;
    omega += alpha * dt;

    yaw = wrap_to_pi(yaw + omega * dt);
    cy = cosf(yaw);
    sy = sinf(yaw);

    vx_world = vx_body * (cp * cy) + vy_body * (sr * sp * cy - cr * sy);
    vy_world = vx_body * (cp * sy) + vy_body * (sr * sp * sy + cr * cy);
    x += vx_world * dt;
    y += vy_world * dt;

    ay_gravity = (vx_body * omega - sr * GRAVITY);
    az = GRAVITY * ct - vx_body * wy - vy_body * wx;

    valid[k] = valid[k] && vx_body > min_vel && vx_body < max_vel;
    valid[k] = valid[k] && vy_body > min_vel && vy_body < max_vel;
    valid[k] = valid[k] && omega > -max_omega && omega < max_omega;
    valid[k] = valid[k] && fabsf(vx_body * wy) < max_vert_acc;
    valid[k] = valid[k] && fabsf(vy_body * wx) < max_vert_acc;
    valid[k] = valid[k] && fabsf(ay_gravity / az) < RI;
    valid[k] = valid[k] && fabsf(pitch) < max_theta && fabsf(roll) < max_theta;

    intermediate_index = k * timesteps * NX + (t - 1) * NX;
    intermediate_states[intermediate_index + x_index] = x;
    intermediate_states[intermediate_index + y_index] = y;
    intermediate_states[intermediate_index + yaw_index] = yaw;
    intermediate_states[intermediate_index + vx_index] = vx_body;
    intermediate_states[intermediate_index + vy_index] = vy_body;
    intermediate_states[intermediate_index + omega_index] = omega;

    if (!valid[k]) {
      break;
    }
  }

  cost[k] = fabsf(timesteps * dt);

  state[state_base + x_index] = x;
  state[state_base + y_index] = y;
  state[state_base + yaw_index] = yaw;
  state[state_base + vx_index] = vx_body;
  state[state_base + vy_index] = vy_body;
  state[state_base + omega_index] = omega;
}

void holonomic_6d_launcher(
    float *state, float *intermediate_states, const TerrainMap &map,
    bool *valid, float *cost, float dt, int timesteps, int n_succ, int NX,
    int NC, float car_l2, float car_w2, float accel_max, float alpha_max,
    float max_vel, float min_vel, float max_omega, float RI, float max_vert_acc,
    float max_theta, int patch_length_px, int patch_width_px, const int blocks,
    const int threads, float *d_state, float *d_intermediate_states,
    float *d_controls, bool *d_valid, float *d_cost, cudaStream_t stream) {
  dim3 valid_threads(patch_length_px, patch_width_px);
  dim3 valid_blocks(timesteps, n_succ);
  cudaMemcpyAsync(d_state, state, sizeof(float) * n_succ * NX,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_valid, valid, n_succ * sizeof(bool), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(d_cost, cost, n_succ * sizeof(float), cudaMemcpyHostToDevice,
                  stream);

  holonomic_6d_kernel<<<blocks, threads, 0, stream>>>(
      d_state, d_intermediate_states, d_controls, map, d_valid, d_cost, dt,
      timesteps, n_succ, NX, NC, car_l2, car_w2, accel_max, alpha_max,
      max_vel, min_vel, max_omega, RI, max_vert_acc, max_theta);
  check_validity_batch_kernel<<<valid_blocks, valid_threads, 0, stream>>>(
      map, d_intermediate_states, patch_length_px, patch_width_px, car_l2,
      car_w2, NX, timesteps, d_valid);
  cudaMemcpyAsync(state, d_state, sizeof(float) * n_succ * NX,
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(valid, d_valid, sizeof(bool) * n_succ, cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(cost, d_cost, sizeof(float) * n_succ, cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(intermediate_states, d_intermediate_states,
                  sizeof(float) * n_succ * timesteps * NX,
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
}

void check_validity_launcher(const TerrainMap &map, float *states,
                             int patch_length_px, int patch_width_px,
                             float car_l2, float car_w2, int n_states, int NX,
                             bool *result) {
  dim3 threads(patch_length_px, patch_width_px);
  dim3 blocks(1, n_states);
  bool *d_result;
  float *d_validity_states;
  cudaMalloc(&d_result, n_states * sizeof(bool));
  cudaMalloc(&d_validity_states, NX * n_states * sizeof(float));
  cudaMemcpy(d_result, result, n_states * sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_validity_states, states, n_states * NX * sizeof(float),
             cudaMemcpyHostToDevice);

  check_validity_batch_kernel<<<blocks, threads>>>(
      map, d_validity_states, patch_length_px, patch_width_px, car_l2, car_w2,
      NX, 1, d_result);

  cudaMemcpy(result, d_result, n_states * sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(d_result);
  cudaFree(d_validity_states);
}
