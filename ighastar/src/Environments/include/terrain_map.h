#pragma once

#include <cuda_runtime.h>

// Terrain representation used by the kinodynamic environment. The planner
// itself is agnostic to this choice: only map_to_elev() and map_to_cost() in
// kinodynamic.cu look at it.
enum MapType {
  MAP_ELEVATION = 0, // elevation map + costmap (2D)
  MAP_ESDF = 1       // signed distance field + colour voxels (3D)
};

// Everything the device queries need to read the world, passed by value into
// the kernels. Swapping the synthetic ESDF for a real (nvblox) one means
// filling these pointers from a different source; no kernel changes.
struct TerrainMap {
  int type = MAP_ELEVATION;

  // Elevation mode: nx * ny grids indexed [iy * nx + ix].
  const float *elev = nullptr;
  const float *cost = nullptr;

  // ESDF mode: nx * ny * nz grids indexed [(iy * nx + ix) * nz + k], so the z
  // axis is contiguous within a column. Signed distance is negative below the
  // terrain surface; the colour layer is white for free and black for
  // obstacle, mirroring nvblox's ColorVoxel layer.
  const float *esdf = nullptr;
  const uchar4 *color = nullptr;

  int nx = 0, ny = 0, nz = 0;
  float voxel_xy = 0.0f; // horizontal resolution, metres per cell
  float voxel_z = 0.0f;  // vertical resolution, metres per voxel
  float z_min = 0.0f;    // world height of ESDF voxel k = 0
  float res_inv = 0.0f;  // 1 / voxel_xy
};
