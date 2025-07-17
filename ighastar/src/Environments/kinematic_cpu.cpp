#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cstdio>

// Maps world coordinates to elevation value from heightmap
float map_to_elev_cpu(float x, float y, const float* elev, int map_size_px, float res_inv) {
    int img_X = std::min(std::max(static_cast<int>(x * res_inv), 0), map_size_px - 1);
    int img_Y = std::min(std::max(static_cast<int>(y * res_inv), 0), map_size_px - 1);
    return elev[img_Y * map_size_px + img_X];
} 

// Computes the 3D footprint coordinates and center height of the car
void get_footprint_z_cpu(float* fl, float* fr, float* bl, float* br, float& z,
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

    float z_cent = map_to_elev_cpu(0, 0, elev, map_size_px, res_inv);
    z = map_to_elev_cpu(x, y, elev, map_size_px, res_inv) - z_cent;

    fl[2] = map_to_elev_cpu(fl[0], fl[1], elev, map_size_px, res_inv) - z_cent;
    fr[2] = map_to_elev_cpu(fr[0], fr[1], elev, map_size_px, res_inv) - z_cent;
    bl[2] = map_to_elev_cpu(bl[0], bl[1], elev, map_size_px, res_inv) - z_cent;
    br[2] = map_to_elev_cpu(br[0], br[1], elev, map_size_px, res_inv) - z_cent;
}

// Checks if a car footprint collides with obstacles in the map
bool check_crop_cpu(const float x, const float y, const float cy, const float sy, 
                    const float* map, const int map_size_px, const float res,
                    const float car_l2, const float car_w2) {
    float res_inv = 1.0f / res;
    int patch_length_px = int(2*car_l2*res_inv);
    int patch_width_px = int(2*car_w2*res_inv);
    
    for(int i = 0; i < patch_length_px; i++) {
        float offset_x = (i*res) - (car_l2);
        for(int j = 0; j < patch_width_px; j++) {
            float offset_y = (j*res) - (car_w2);
            float px = offset_x*cy - offset_y*sy + x;
            float py = offset_x*sy + offset_y*cy + y;
            
            if (px < 0 || px >= map_size_px*res || py < 0 || py >= map_size_px*res) {
                return false;
            }
            
            int img_X = std::min(std::max(static_cast<int>(px * res_inv), 0), map_size_px - 1);
            int img_Y = std::min(std::max(static_cast<int>(py * res_inv), 0), map_size_px - 1);
            
            if (map[img_Y * map_size_px + img_X] < 250.0f) {
                return false;
            }
        }
    }
    return true;
}

// Launches kinematic simulation for multiple rollouts on CPU
void kinematic_launcher_cpu(
    float* state, float* intermediate_states, const float* heightmap, const float* costmap, 
    bool* valid, float* cost, float dt, int timesteps, int rollouts, int n_dims, int n_cont,
    const int map_size_px, float map_res,
    float car_l2, float car_w2, float max_vel, float min_vel, float RI,
    float max_vert_acc, float max_theta, float gear_switch_time, int patch_length_px, int patch_width_px,
    const int blocks, const int threads, const float* controls) {
    
    // CPU implementation based on analytical_bicycle.cpp kinematic_launcher
    float res_inv = 1.0f / map_res;
    
    for (int k = 0; k < rollouts; k++) {
        int state_base = k * n_dims;
        int control_base = k * n_cont;
        
        // Extract current state
        float x = state[state_base + 0];
        float y = state[state_base + 1];
        float yaw = state[state_base + 2];
        
        // Extract controls from controls array
        float curvature = controls[control_base + 0];  // steering
        float vx = controls[control_base + 1];   // throttle/wheelspeed
        float wz = curvature * vx;  // angular velocity
        
        // Additional variables for 3D motion (like CUDA version)
        float vy = 0, vz = 0;
        
        valid[k] = true;
        cost[k] = 0.0f;
        
        // Compute initial footprint & orientation (exactly like CUDA version)
        float cy = cosf(yaw), sy = sinf(yaw);
        float fl[3], fr[3], bl[3], br[3], z;
        float roll, pitch, cp, sp, cr, sr;
        valid[k] = true;

        for (int t = 1; t <= timesteps; t++) {
            cy = cosf(yaw); sy = sinf(yaw);
            get_footprint_z_cpu(fl, fr, bl, br, z, x, y, cy, sy, heightmap, map_size_px, res_inv, car_l2, car_w2);
            roll = atan2f((fl[2] + bl[2]) - (fr[2] + br[2]), 4 * car_w2);
            pitch = atan2f((bl[2] + br[2]) - (fl[2] + fr[2]), 4 * car_l2);
            
            cp = cosf(pitch), sp = sinf(pitch);
            cr = cosf(roll), sr = sinf(roll);

            yaw = fmod(yaw + wz * dt + M_PI, 2 * M_PI) - M_PI;  // wrap_to_pi equivalent
            cy = cosf(yaw); sy = sinf(yaw);

            x += dt * (vx * (cp * cy) + vy * (sr * sp * cy - cr * sy) + vz * (cr * sp * cy + sr * sy));
            y += dt * (vx * (cp * sy) + vy * (sr * sp * sy + cr * cy) + vz * (cr * sp * sy - sr * cy));

            // Check constraints
            if (valid[k]) {
                valid[k] = fabsf(pitch) < max_theta && fabsf(roll) < max_theta;
                valid[k] = valid[k] && check_crop_cpu(x, y, cy, sy, costmap, map_size_px, map_res, car_l2, car_w2);
            }
            
            // Store intermediate state
            int intermediate_base = k * timesteps * n_dims + (t-1) * n_dims;
            intermediate_states[intermediate_base + 0] = x;
            intermediate_states[intermediate_base + 1] = y;
            intermediate_states[intermediate_base + 2] = yaw;
            intermediate_states[intermediate_base + 3] = vx;
        }
        float gear_switch_cost = gear_switch_time * (vx < 0); // reverse cost
        cost[k] = timesteps * dt * fabs(vx) + gear_switch_cost;
        
        // Update final state
        state[state_base + 0] = x;
        state[state_base + 1] = y;
        state[state_base + 2] = yaw;
        state[state_base + 3] = vx;
    }
}

// Checks validity of multiple states against a bitmap on CPU
void check_validity_launcher_cpu(
    const float* bitmap, int map_size_px, float map_res,
    float* states,
    int patch_length_px, int patch_width_px,
    float car_l2, float car_w2, int n_states, int n_dims,
    bool* result) {
    
    for (int i = 0; i < n_states; i++) {
        float x = states[i * n_dims + 0];
        float y = states[i * n_dims + 1];
        float yaw = states[i * n_dims + 2];
        
        float cy = cosf(yaw);
        float sy = sinf(yaw);
        
        result[i] = check_crop_cpu(x, y, cy, sy, bitmap, map_size_px, map_res, car_l2, car_w2);
    }
}