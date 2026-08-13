// Flatten a dense colored ESDF into a fixed-size BEV elevation map + costmap.
// Independent of IGHA*: one thread per BEV cell, no planner state.
//
// Layout (same as the synthetic ESDF / TerrainMap ESDF mode):
//   distance[iy, ix, k] with z contiguous in a column:
//   index = (iy * width + ix) * nz + k
// Color may be either:
//   3D [H, W, nz, 3] — sample at the surface voxel, or
//   2D [H, W, 3]     — column-constant (synthetic ESDF / cheaper upload)
// Surface height is the zero crossing of the signed distance (negative below
// terrain). Cost is the BT.601 luminance of the colour, on the planner's
// 0-255 scale (white free, black obstacle).

#include <cuda_runtime.h>

#include <cstdint>

namespace esdf_bev {

__device__ inline float clampf(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}

__device__ inline int clampi(int x, int lo, int hi) {
  return max(lo, min(x, hi));
}

// Recover the terrain surface in one ESDF column. Finds the topmost voxel at
// or below the surface (distance <= 0) and linearly interpolates against the
// free-space sample above it — same rule as esdf_utils.surface_grid.
__device__ float column_surface_z(const float *distance, int nz, float voxel_z,
                                  float z_min) {
  int k_lo = 0;
  for (int k = nz - 1; k >= 0; k--) {
    if (distance[k] <= 0.0f) {
      k_lo = k;
      break;
    }
  }
  int k_hi = min(k_lo + 1, nz - 1);
  float d_lo = distance[k_lo];
  float d_hi = distance[k_hi];
  float delta = d_hi - d_lo;
  float frac = (fabsf(delta) > 1e-12f) ? (-d_lo / delta) : 0.0f;
  frac = clampf(frac, 0.0f, 1.0f);
  return z_min + (float(k_lo) + frac) * voxel_z;
}

__global__ void esdf_to_bev_kernel(const float *distance, const uint8_t *color,
                                   float *elev, float *cost, int height,
                                   int width, int nz, int color_nz,
                                   float voxel_z, float z_min) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix >= width || iy >= height)
    return;

  const int cell = iy * width + ix;
  const float *col_d = distance + cell * nz;
  float z = column_surface_z(col_d, nz, voxel_z, z_min);

  const uint8_t *rgb;
  if (color_nz <= 1) {
    rgb = color + cell * 3;
  } else {
    int k = clampi(static_cast<int>((z - z_min) / voxel_z), 0, nz - 1);
    rgb = color + (cell * nz + k) * 3;
  }
  float luminance =
      0.299f * float(rgb[0]) + 0.587f * float(rgb[1]) + 0.114f * float(rgb[2]);

  elev[cell] = z;
  cost[cell] = luminance;
}

void launch_esdf_to_bev(const float *distance, const uint8_t *color,
                        float *elev, float *cost, int height, int width, int nz,
                        int color_nz, float voxel_z, float z_min) {
  dim3 threads(32, 32);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);
  esdf_to_bev_kernel<<<blocks, threads>>>(distance, color, elev, cost, height,
                                          width, nz, color_nz, voxel_z, z_min);
}

} // namespace esdf_bev
