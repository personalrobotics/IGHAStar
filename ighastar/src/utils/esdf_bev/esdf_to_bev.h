#pragma once

#include <cstdint>

namespace esdf_bev {

// color_nz == 1 means a column-constant [H, W, 3] colour map; otherwise colour
// is [H, W, nz, 3] and is sampled at the recovered surface voxel.
void launch_esdf_to_bev(const float *distance, const uint8_t *color,
                        float *elev, float *cost, int height, int width, int nz,
                        int color_nz, float voxel_z, float z_min);

} // namespace esdf_bev
