#include "terrain_map.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iomanip>

#define x_index 0
#define y_index 1
#define yaw_index 2
#define vx_index 3

#define st_index 0
#define th_index 1
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

// Maximum number of steps the ESDF column march is allowed to take. The march
// makes at least one voxel of progress per step, and typically converges in
// two or three, so this is only a safety net.
#define MAX_MARCH_STEPS 32

// Index of the ESDF column holding world coordinates (x, y) (device function)
__device__ int esdf_column(float x, float y, const TerrainMap &map) {
  int img_X = clamp(static_cast<int>((x * map.res_inv)), 0, map.nx - 1);
  int img_Y = clamp(static_cast<int>((y * map.res_inv)), 0, map.ny - 1);
  return (img_Y * map.nx + img_X) * map.nz;
}

// Finds the terrain surface in an ESDF column: march down from the top of the
// column, stepping by the signed distance (which never exceeds the vertical
// gap to the surface, so we cannot step past it), then linearly interpolate
// between the two voxels straddling the zero crossing (device function).
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

// Maps world coordinates to terrain height, from whichever representation the
// world was loaded with (device function)
__device__ float map_to_elev(float x, float y, const TerrainMap &map) {
  if (map.type == MAP_ELEVATION) {
    int img_X = clamp(static_cast<int>((x * map.res_inv)), 0, map.nx - 1);
    int img_Y = clamp(static_cast<int>((y * map.res_inv)), 0, map.ny - 1);
    return map.elev[img_Y * map.nx + img_X];
  }
  return esdf_surface_z(x, y, map);
}

// Maps world coordinates to a traversability value on the same 0-255 scale as
// the costmap: 255 is free, 0 is an obstacle (device function)
__device__ float map_to_cost(float x, float y, const TerrainMap &map) {
  if (map.type == MAP_ELEVATION) {
    int img_X = clamp(static_cast<int>((x * map.res_inv)), 0, map.nx - 1);
    int img_Y = clamp(static_cast<int>((y * map.res_inv)), 0, map.ny - 1);
    return map.cost[img_Y * map.nx + img_X];
  }
  // ESDF: the colour of the voxel at the recovered surface, as luminance.
  const int column = esdf_column(x, y, map);
  int k = clamp(static_cast<int>((esdf_surface_z(x, y, map) - map.z_min) /
                                 map.voxel_z),
                0, map.nz - 1);
  uchar4 c = map.color[column + k];
  return 0.299f * float(c.x) + 0.587f * float(c.y) + 0.114f * float(c.z);
}

// Computes the 3D footprint coordinates and center height of the car (device
// function)
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

/*
list of constants:
    NX = 4
    NC = 2
    timesteps = 10
    n_succ = 1000
    patch_length_px = 20
    patch_width_px = 20
    map_size_px = 1000
    map_res = 0.1
    car_l2 = 1.0
    car_w2 = 0.5
*/
/*
list of reused variables:
    map = TerrainMap (costmap or ESDF colour layer)
    d_intermediate_states = d_intermediate_states
    patch_length_px = patch_length_px
    patch_width_px = patch_width_px
    car_l2 = car_l2
    car_w2 = car_w2
    d_valid = d_valid
*/

// Checks validity of multiple states against the traversability layer (CUDA
// kernel)
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
    return; // already invalid

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

// Launches kinodynamic simulation for multiple rollouts (CUDA kernel)
__global__ void kinodynamic_kernel(float *state, float *intermediate_states,
                                   float *controls, TerrainMap map, bool *valid,
                                   float *cost, float dt, int timesteps,
                                   int rollouts, int NX, int NC, float car_l2,
                                   float car_w2, float max_vel, float min_vel,
                                   float RI, float max_vert_acc,
                                   float max_theta, float gear_switch_time) {
  int k = blockIdx.x * blockDim.x + threadIdx.x; // rollout ID

  if (k >= rollouts)
    return;

  int state_base = k * NX;
  int intermediate_index;

  float x = state[state_base + x_index];
  float y = state[state_base + y_index];
  float yaw = state[state_base + yaw_index];
  float vx = state[state_base + vx_index];
  float vz = 0.0f, vy = 0.0f;
  float wz = 0.0f;

  // Compute initial footprint & orientation
  float cy = cosf(yaw), sy = sinf(yaw);
  float fl[3], fr[3], bl[3], br[3], z;
  get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, map, car_l2, car_w2);
  float last_roll = atan2f((fl[2] + bl[2]) - (fr[2] + br[2]), 4 * car_w2);
  float last_pitch = atan2f((bl[2] + br[2]) - (fl[2] + fr[2]), 4 * car_l2);
  float roll, pitch, wx, wy, cp, sp, cr, sr, ay, az;
  float initial_vx = vx;
  valid[k] = true;

  for (int t = 1; t <= timesteps; t++) {
    int control_base = k * timesteps * NC + (t - 1) * NC;
    float curvature = controls[control_base + st_index];
    float ax = controls[control_base + th_index];
    wz = curvature * vx;

    cy = cosf(yaw);
    sy = sinf(yaw);
    get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, map, car_l2, car_w2);
    roll = atan2f((fl[2] + bl[2]) - (fr[2] + br[2]), 4 * car_w2);
    pitch = atan2f((bl[2] + br[2]) - (fl[2] + fr[2]), 4 * car_l2);

    wx = (roll - last_roll) / dt;
    wy = (pitch - last_pitch) / dt;
    last_pitch = pitch;
    last_roll = roll;

    cp = cosf(pitch), sp = sinf(pitch);
    cr = cosf(roll), sr = sinf(roll);

    vx += (ax * cp + sp * GRAVITY) * dt;
    ay = vx * wz - sr * GRAVITY;
    az = GRAVITY * cp * cr - vx * wy +
         wx * wx * car_w2; // assuming car width/2 ~ car cg height
    wz = curvature * vx;

    yaw = wrap_to_pi(yaw + wz * dt);
    cy = cosf(yaw);
    sy = sinf(yaw);

    x += dt * (vx * (cp * cy) + vy * (sr * sp * cy - cr * sy) +
               vz * (cr * sp * cy + sr * sy));
    y += dt * (vx * (cp * sy) + vy * (sr * sp * sy + cr * cy) +
               vz * (cr * sp * sy - sr * cy));

    valid[k] = valid[k] && fabsf(az - GRAVITY) < max_vert_acc;
    valid[k] = valid[k] && fabsf(ay / az) < RI;
    valid[k] = valid[k] && vx > min_vel && vx < max_vel;
    valid[k] = valid[k] && fabsf(pitch) < max_theta && fabsf(roll) < max_theta;

    intermediate_index = k * timesteps * NX + (t - 1) * NX;
    intermediate_states[intermediate_index + x_index] = x;
    intermediate_states[intermediate_index + y_index] = y;
    intermediate_states[intermediate_index + yaw_index] = yaw;
    intermediate_states[intermediate_index + vx_index] = vx;
  }
  float gear_switch_cost =
      gear_switch_time * (vx * initial_vx < 0); // change in direction
  cost[k] = fabsf(timesteps * dt) + gear_switch_cost;

  state[state_base + x_index] = x;
  state[state_base + y_index] = y;
  state[state_base + yaw_index] = yaw;
  state[state_base + vx_index] = vx;
}

void kinodynamic_launcher(
    float *state, float *intermediate_states, const TerrainMap &map,
    bool *valid, float *cost, float dt, int timesteps, int n_succ, int NX,
    int NC, float car_l2, float car_w2, float max_vel, float min_vel, float RI,
    float max_vert_acc, float max_theta, float gear_switch_time,
    int patch_length_px, int patch_width_px, const int blocks,
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
  // controls layout: [rollout][timestep][n_cont] — caller must upload before launch

  kinodynamic_kernel<<<blocks, threads, 0, stream>>>(
      d_state, d_intermediate_states, d_controls, map, d_valid, d_cost, dt,
      timesteps, n_succ, NX, NC, car_l2, car_w2, max_vel, min_vel, RI,
      max_vert_acc, max_theta, gear_switch_time);
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