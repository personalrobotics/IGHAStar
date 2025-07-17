// my_extension.cpp
#include <torch/extension.h>
#include <cmath>
#include <algorithm>
#include <cfloat>
// #include <omp.h>

#define x_index 0
#define y_index 1
#define yaw_index 2
#define vx_index 3
#define wz_index 4
#define st_index 0
#define th_index 1
#define GRAVITY 9.81f

inline float nan_to_num(float x, float replace) {
    return (std::isnan(x) || std::isinf(x)) ? replace : x;
}

inline float clamp(float x, float lower, float upper) {
    return std::min(std::max(x, lower), upper);
}

inline float map_to_elev(float x, float y, const float* elev, int map_size_px, float res_inv) {
    int img_X = clamp(static_cast<int>((x * res_inv)), 0, map_size_px - 1);
    int img_Y = clamp(static_cast<int>((y * res_inv)), 0, map_size_px - 1);
    return elev[img_Y * map_size_px + img_X];
}

void get_footprint_z(float* fl, float* fr, float* bl, float* br, float& z,
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

bool check_crop(const float x, const float y, const float cy, const float sy, 
                        const float* map, const int map_size_px, const float res,
                        const float car_l2, const float car_w2)
{
    float res_inv = 1.0f / res;
    int img_X, img_Y, patch_length_px = int(2*car_l2*res_inv), patch_width_px = int(2*car_w2*res_inv);
    float px, py, offset_x, offset_y;
    bool valid = true;
    for(int i=0; i < patch_length_px; i++)
    {
        offset_x = (i*res) - (car_l2);
        for(int j=0; j < patch_width_px; j++)
        {
            offset_y = (j*res) - (car_w2);
            px = offset_x*cy - offset_y*sy + x;
            py = offset_x*sy + offset_y*cy + y;
            valid = px > 0 && px < map_size_px*res && py > 0 && py < map_size_px*res && map_to_elev(px, py, map, map_size_px, res_inv) > 250.0;
            if(!valid)
            {
                return false;
            }
        }
    }
    return true;
}

bool check_validity(const float x, const float y, const float cy, const float sy, 
                    torch::Tensor map_tensor, const int map_size_px, const float res,
                    const float car_l2, const float car_w2)
{
    auto map = map_tensor.contiguous().data_ptr<float>();
    return check_crop(x, y, cy, sy, map, map_size_px, res, car_l2, car_w2);
}

inline float wrap_to_pi(float x) {
    return std::fmod(x + M_PI, 2 * M_PI) - M_PI;
}

void kinematic_launcher(torch::Tensor state_tensor, torch::Tensor controls_tensor, torch::Tensor BEVmap_height_tensor, torch::Tensor BEVmap_cost_tensor, torch::Tensor valid_tensor,
                        torch::Tensor cost_tensor, float dt, int rollouts, int timesteps, int NX, int NC, float throttle_to_wheelspeed, float steering_max, int BEVmap_size_px, float BEVmap_res,
                        float car_l2, float car_w2) {
    auto state = state_tensor.contiguous().data_ptr<float>();
    auto controls = controls_tensor.contiguous().data_ptr<float>();
    auto BEVmap_height = BEVmap_height_tensor.contiguous().data_ptr<float>();
    auto BEVmap_cost = BEVmap_cost_tensor.contiguous().data_ptr<float>();
    auto valid = valid_tensor.contiguous().data_ptr<bool>();
    auto cost = cost_tensor.contiguous().data_ptr<float>();

    // omp requires these to be within the for loop
    float z;
    float st, th;
    float x, y, yaw, vx, wz;
    int t = 0, state_base, control_base;
    float w, K;
    float cy, sy;
    float res_inv = 1.0f / BEVmap_res;

    // NX = 3, NC = 2. We have timesteps but we  don't have them in the "state"
    // the state array is rollouts x NX. The initial value of all the states is the same as the start node.
    // omp_set_num_threads(rollouts);
    // #pragma omp parallel for
    float start_yaw = state[yaw_index];
    for (int k = 0; k < rollouts; k++) 
    {

        state_base = k * NX;
        control_base = k * NC;
        // asume constant controls for all timesteps
        st = controls[control_base + st_index] * steering_max;
        w = controls[control_base + th_index] * throttle_to_wheelspeed;
        K = tan(st) / (car_l2 * 2);

        x = state[state_base + x_index];
        y = state[state_base + y_index];
        yaw = state[state_base + yaw_index];

        vx = w;
        wz = K * w;
        for (t = 0; t < timesteps; ++t) {
            yaw = wrap_to_pi(yaw + wz * dt);
            cy = cos(yaw);
            sy = sin(yaw);

            x += dt * vx * cy;
            y += dt * vx * sy;
            if(valid[k] && !check_crop(x, y, cy, sy, BEVmap_cost, BEVmap_size_px, BEVmap_res, car_l2, car_w2))
            {
                valid[k] = false;
                break;
            }
        }
        cost[k] = abs(vx) * timesteps * dt;
        state[state_base + x_index] = x;
        state[state_base + y_index] = y;
        state[state_base + yaw_index] = yaw;
    }
    // #pragma omp barrier
}

void kinodynamic_launcher(torch::Tensor state_tensor, torch::Tensor controls_tensor, torch::Tensor BEVmap_height_tensor, torch::Tensor BEVmap_cost_tensor, torch::Tensor valid_tensor,
                        torch::Tensor cost_tensor, float dt, int rollouts, int timesteps, int NX, int NC, float throttle_to_accel, float steering_max, int BEVmap_size_px, float BEVmap_res,
                        float car_l2, float car_w2, float max_vel, float min_vel, float RI, float max_vert_acc, float max_theta) {
    auto state = state_tensor.contiguous().data_ptr<float>();
    auto controls = controls_tensor.contiguous().data_ptr<float>();
    auto BEVmap_height = BEVmap_height_tensor.contiguous().data_ptr<float>();
    auto BEVmap_cost = BEVmap_cost_tensor.contiguous().data_ptr<float>();
    auto valid = valid_tensor.contiguous().data_ptr<bool>();
    auto cost = cost_tensor.contiguous().data_ptr<float>();

    float res_inv = 1.0f / BEVmap_res;

    float fl[3], fr[3], bl[3], br[3];
    float z;
    float st, th;
    float vx, vy, vz, ax, ay, az, wx, wy, wz;
    float cp, sp, cr, sr, ct, cy, sy;
    float x, y, roll, pitch, yaw;
    int t = 0, state_base, control_base;
    float w, K;
    float last_pitch, last_roll, last_vx;
    // NX = 3, NC = 2. We have timesteps but we  don't have them in the "state"
    // the state array is rollouts x NX. The initial value of all the states is the same as the start node.

    x = state[0 + x_index];
    y = state[0 + y_index];
    yaw = state[0 + yaw_index];
    vx = state[0 + vx_index];
    wz = state[0 + wz_index];

    cy = cosf(yaw);
    sy = sinf(yaw);
    get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, BEVmap_height, BEVmap_size_px, res_inv, car_l2, car_w2);
    float initial_roll = std::atan2((fl[2] + bl[2]) - (fr[2] + br[2]), 4 * car_w2);
    float initial_pitch = std::atan2((bl[2] + br[2]) - (fl[2] + fr[2]), 4 * car_l2);

    for (int k = 0; k < rollouts; ++k) {
        state_base = k * NX;
        control_base = k * NC;
        // asume constant controls for all timesteps
        st = controls[control_base + st_index] * steering_max;
        K = std::tan(st) / (car_l2 * 2);

        x = state[state_base + x_index];
        y = state[state_base + y_index];
        yaw = state[state_base + yaw_index];
        vx = state[state_base + vx_index];
        ax = controls[control_base + th_index] * throttle_to_accel;
        vy = 0;
        vz = 0;
        wz = K * vx;
        last_roll = initial_roll;
        last_pitch = initial_pitch;
        last_vx = vx;
        valid[k] = true;

        for (t = 0; t < timesteps; ++t) {
            cy = cosf(yaw);
            sy = sinf(yaw);

            get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, BEVmap_height, BEVmap_size_px, res_inv, car_l2, car_w2);
            roll = std::atan2((fl[2] + bl[2]) - (fr[2] + br[2]), 4 * car_w2);
            pitch = std::atan2((bl[2] + br[2]) - (fl[2] + fr[2]), 4 * car_l2);
            
            wx = (roll - last_roll) / dt;
            wy = (pitch - last_pitch) / dt;
            last_pitch = pitch;
            last_roll = roll;

            cp = cosf(pitch);
            sp = sinf(pitch);
            cr = cosf(roll);
            sr = sinf(roll);
            ct = nan_to_num(std::sqrt(1 - (sp * sp) - (sr * sr)), 0.0);

            vx += (ax + sp * GRAVITY) * dt;
            vx = clamp(vx, min_vel, max_vel);
            ay = (vx * wz - sr * GRAVITY);
            az = GRAVITY * ct - vx * wy;
            wz = K * vx;

            yaw = wrap_to_pi(yaw + wz * dt);
            cy = cosf(yaw);
            sy = sinf(yaw);

            x += dt * (vx * (cp * cy) + vy * (sr * sp * cy - cr * sy) + vz * (cr * sp * cy + sr * sy));
            y += dt * (vx * (cp * sy) + vy * (sr * sp * sy + cr * cy) + vz * (cr * sp * sy - sr * cy));
            valid[k] = valid[k] && check_crop(x, y, cy, sy, BEVmap_cost, BEVmap_size_px, BEVmap_res, car_l2, car_w2);
            valid[k] = valid[k] && abs(vx * wy) < max_vert_acc;
            valid[k] = valid[k] && abs(ay/az) < RI;
            valid[k] = valid[k] && abs(pitch) < max_theta && abs(roll) < max_theta;
            if(!valid[k])
            {
                break;
            }
        }
        cost[k] = timesteps * dt;
        state[state_base + x_index] = x;
        state[state_base + y_index] = y;
        state[state_base + yaw_index] = yaw;
        state[state_base + vx_index] = vx;
        state[state_base + wz_index] = 0.0f;
    }
}

// PyTorch bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rollout_kinematic", &kinematic_launcher, "kinematic rollout");
    m.def("rollout_kinodynamic", &kinodynamic_launcher, "kinodynamic rollout");
    m.def("check_crop", &check_validity, "check crop");
}