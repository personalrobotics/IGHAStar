#include <cmath>
#include <algorithm>
#include <cfloat>
#include <iomanip>

#define x_index 0
#define y_index 1
#define yaw_index 2
#define vx_index 3

#define st_index 0
#define th_index 1
#define GRAVITY 9.81f

// using namespace std;
float *d_controls, *d_state,  *d_cost, *d_intermediate_states;
bool *d_valid;

// Sets up CUDA memory for kinematic simulation
void cuda_setup(float* controls, int n_succ, int NX, int NC, int timesteps)
{
    cudaMalloc(&d_controls, sizeof(float) * n_succ * NC);
    cudaMalloc(&d_state, n_succ * NX * sizeof(float));
    cudaMalloc(&d_intermediate_states, n_succ * timesteps * NX * sizeof(float));
    cudaMalloc(&d_valid, n_succ * sizeof(bool));
    cudaMalloc(&d_cost, n_succ * sizeof(float));
    cudaMemcpy(d_controls, controls, sizeof(float) * n_succ * NC, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

// Cleans up CUDA memory for kinematic simulation
void cuda_cleanup()
{
    cudaFree(d_controls);
    cudaFree(d_state);
    cudaFree(d_intermediate_states);
    cudaFree(d_valid);
    cudaFree(d_cost);
}

__device__ float nan_to_num(float x, float replace) {
    return (std::isnan(x) || std::isinf(x)) ? replace : x;
}

__device__ float clamp(float x, float lower, float upper) {
    return std::min(std::max(x, lower), upper);
}

__device__ float wrap_to_pi(float x) {
    return std::fmod(x + M_PI, 2 * M_PI) - M_PI;
}

// Maps world coordinates to elevation value from heightmap (device function)
__device__ float map_to_elev(float x, float y, const float* elev, int map_size_px, float res_inv) {
    int img_X = clamp(static_cast<int>((x * res_inv)), 0, map_size_px - 1);
    int img_Y = clamp(static_cast<int>((y * res_inv)), 0, map_size_px - 1);
    return elev[img_Y * map_size_px + img_X];
}

// Computes the 3D footprint coordinates and center height of the car (device function)
__device__ void get_footprint_z(float* fl, float* fr, float* bl, float* br, float& z,
                     float x, float y, float cy, float sy,
                     const float* elev, int map_size_px, float res_inv,
                     float car_l2, float car_w2) {
    fl[0] = car_l2 * cy - car_w2 * sy + x;
    fl[1] = car_l2 * sy + car_w2 * cy + y;

    fr[0] = car_l2 * cy - (-1) * car_w2 * sy + x;
    fr[1] = car_l2 * sy + (-1) * car_w2 * cy + y;

    bl[0] = (-1) * car_l2 * cy - car_w2 * sy + x;
    bl[1] = (-1) * car_l2 * sy + car_w2 * cy + y;

    br[0] = (-1) * car_l2 * cy - (-1) * car_w2 * sy + x;
    br[1] = (-1) * car_l2 * sy + (-1) * car_w2 * cy + y;

    float z_cent = map_to_elev(0, 0, elev, map_size_px, res_inv);
    z = map_to_elev(x, y, elev, map_size_px, res_inv) - z_cent;

    fl[2] = map_to_elev(fl[0], fl[1], elev, map_size_px, res_inv) - z_cent;
    fr[2] = map_to_elev(fr[0], fr[1], elev, map_size_px, res_inv) - z_cent;
    bl[2] = map_to_elev(bl[0], bl[1], elev, map_size_px, res_inv) - z_cent;
    br[2] = map_to_elev(br[0], br[1], elev, map_size_px, res_inv) - z_cent;
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
    bitmap = BEVmap_cost
    map_size_px = map_size
    map_res = map_res
    d_intermediate_states = d_intermediate_states
    patch_length_px = patch_length_px
    patch_width_px = patch_width_px
    car_l2 = car_l2
    car_w2 = car_w2
    d_valid = d_valid
*/

// Checks validity of multiple states against a bitmap (CUDA kernel)
__global__ void check_validity_batch_kernel(
        const float* bitmap, int map_size_px, float map_res,
        float* d_intermediate_states,
        int patch_length_px, int patch_width_px,
        float car_l2, float car_w2, int NX, int timesteps,
        bool* d_valid
    ) {
    int t = blockIdx.x;
    int k = blockIdx.y;
    int i = threadIdx.x;
    int j = threadIdx.y;

    if (i >= patch_length_px || j >= patch_width_px) return;
    if (!(d_valid[k])) return;  // already invalid

    int intermediate_index = k * timesteps * NX + (t - 1) * NX; // NX, NC, timesteps, n_succ are constants. Why are we passing them in constantly? TODO: set up constants separately.

    float x = d_intermediate_states[intermediate_index + x_index];
    float y = d_intermediate_states[intermediate_index + y_index];
    float theta = d_intermediate_states[intermediate_index + yaw_index];

    float cy = cosf(theta);
    float sy = sinf(theta);
    float offset_x = (i * map_res) - car_l2;
    float offset_y = (j * map_res) - car_w2;

    float px = offset_x * cy - offset_y * sy + x;
    float py = offset_x * sy + offset_y * cy + y;

    if (px < 0 || px >= map_size_px * map_res || py < 0 || py >= map_size_px * map_res) {
        d_valid[k] = false;
        return;
    }

    if (map_to_elev(px, py, bitmap, map_size_px, 1.0f / map_res) <= 250.0f) {
        d_valid[k] = false;
    }

}

// Launches kinematic simulation for multiple rollouts (CUDA kernel)
__global__ void kinematic_kernel(
    float* state, float* intermediate_states, float* controls, const float* BEVmap_height, const float* BEVmap_cost, 
    bool* valid, float* cost, float dt, int timesteps, int rollouts, int NX, int NC,
    const int BEVmap_size_px, float BEVmap_res,
    float car_l2, float car_w2, float max_vel, float min_vel, float RI,
    float max_vert_acc, float max_theta, float gear_switch_time)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x; // rollout ID

    if (k >= rollouts) return;

    // Constants
    float res_inv = 1.0f / BEVmap_res;

    int state_base = k * NX;
    int control_base = k * NC;
    int intermediate_index;

    float curvature = controls[control_base + st_index];
    float vx = controls[control_base + th_index];

    float x = state[state_base + x_index];
    float y = state[state_base + y_index];
    float yaw = state[state_base + yaw_index];
    float wz = curvature * vx;
    float vy = 0, vz = 0;

    // Compute initial footprint & orientation
    float cy = cosf(yaw), sy = sinf(yaw);
    float fl[3], fr[3], bl[3], br[3], z;
    float roll, pitch, cp, sp, cr, sr;
    valid[k] = true;

    for (int t = 1; t <= timesteps; t++) {
        cy = cosf(yaw); sy = sinf(yaw);
        get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, BEVmap_height, BEVmap_size_px, res_inv, car_l2, car_w2);
        roll = atan2f((fl[2] + bl[2]) - (fr[2] + br[2]), 4 * car_w2);
        pitch = atan2f((bl[2] + br[2]) - (fl[2] + fr[2]), 4 * car_l2);
        
        cp = cosf(pitch), sp = sinf(pitch);
        cr = cosf(roll), sr = sinf(roll);

        yaw = wrap_to_pi(yaw + wz * dt);
        cy = cosf(yaw); sy = sinf(yaw);

        x += dt * (vx * (cp * cy) + vy * (sr * sp * cy - cr * sy) + vz * (cr * sp * cy + sr * sy));
        y += dt * (vx * (cp * sy) + vy * (sr * sp * sy + cr * cy) + vz * (cr * sp * sy - sr * cy));

        valid[k] = valid[k] && fabsf(pitch) < max_theta && fabsf(roll) < max_theta;

        intermediate_index = k * timesteps * NX + (t - 1) * NX;
        intermediate_states[intermediate_index + x_index] = x;
        intermediate_states[intermediate_index + y_index] = y;
        intermediate_states[intermediate_index + yaw_index] = yaw;
        intermediate_states[intermediate_index + vx_index] = vx;
    }
    float gear_switch_cost = gear_switch_time * (vx < 0); // reverse cost
    cost[k] = timesteps * dt * fabs(vx) + gear_switch_cost;
    state[state_base + x_index] = x;
    state[state_base + y_index] = y;
    state[state_base + yaw_index] = yaw;
    state[state_base + vx_index] = vx;
}

void kinematic_launcher(
    float* state, float *intermediate_states, const float* heightmap, const float* costmap,
    bool* valid, float* cost, float dt, int timesteps, int n_succ, int NX, int NC,
    const int map_size, float map_res,
    float car_l2, float car_w2, float max_vel, float min_vel, float RI,
    float max_vert_acc, float max_theta, float gear_switch_time, int patch_length_px, int patch_width_px, 
    const int blocks, const int threads
) {
    dim3 valid_threads(patch_length_px, patch_width_px);
    dim3 valid_blocks(timesteps, n_succ);
    cudaMemcpy(d_state, state, sizeof(float) * n_succ * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid, valid, n_succ * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost, cost, n_succ * sizeof(float), cudaMemcpyHostToDevice);

    kinematic_kernel<<<blocks, threads>>>(
        d_state, d_intermediate_states, d_controls, heightmap, costmap, 
        d_valid, d_cost, dt, timesteps, n_succ, NX, NC,
        map_size, map_res,
        car_l2, car_w2, max_vel, min_vel, RI,
        max_vert_acc, max_theta, gear_switch_time
    );
    cudaDeviceSynchronize();
    check_validity_batch_kernel<<<valid_blocks, valid_threads>>>(
        costmap, map_size, map_res,
        d_intermediate_states,
        patch_length_px, patch_width_px,
        car_l2, car_w2, NX, timesteps,
        d_valid
    );
    cudaDeviceSynchronize();
    cudaMemcpy(state, d_state, sizeof(float) * n_succ * NX, cudaMemcpyDeviceToHost);
    cudaMemcpy(valid, d_valid, sizeof(bool) * n_succ, cudaMemcpyDeviceToHost);
    cudaMemcpy(cost, d_cost, sizeof(float) * n_succ, cudaMemcpyDeviceToHost);
    cudaMemcpy(intermediate_states, d_intermediate_states, sizeof(float) * n_succ * timesteps * NX, cudaMemcpyDeviceToHost);
}

void check_validity_launcher(
    const float* costmap, int map_size_px, float map_res,
    float* states,
    int patch_length_px, int patch_width_px,
    float car_l2, float car_w2, int n_states, int NX,
    bool* result
) {
    dim3 threads(patch_length_px, patch_width_px);
    dim3 blocks(1, n_states);
    bool* d_result;
    float* d_validity_states;
    cudaMalloc(&d_result, n_states*sizeof(bool));
    cudaMalloc(&d_validity_states, NX * n_states * sizeof(float));
    cudaMemcpy(d_result, &result, n_states*sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_validity_states, states, n_states * NX * sizeof(float), cudaMemcpyHostToDevice);

    check_validity_batch_kernel<<<blocks, threads>>>(
        costmap, map_size_px, map_res,
        d_validity_states,
        patch_length_px, patch_width_px,
        car_l2, car_w2, NX, 1,
        d_result
    );

    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaFree(d_validity_states);
}