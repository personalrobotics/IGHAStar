#include "esdf_to_bev.h"

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <string>
#include <vector>

std::vector<torch::Tensor> esdf_to_bev_cuda(torch::Tensor distance,
                                            torch::Tensor color, float voxel_z,
                                            float z_min) {
  TORCH_CHECK(distance.is_cuda(), "distance must be a CUDA tensor");
  TORCH_CHECK(color.is_cuda(), "color must be a CUDA tensor");
  TORCH_CHECK(distance.dtype() == torch::kFloat32, "distance must be float32");
  TORCH_CHECK(color.dtype() == torch::kUInt8, "color must be uint8");
  TORCH_CHECK(distance.dim() == 3, "distance must be [H, W, nz]");
  TORCH_CHECK(voxel_z > 0.0f, "voxel_z must be positive");
  TORCH_CHECK(distance.size(2) > 1, "nz must be at least 2");

  int color_nz = 1;
  if (color.dim() == 3) {
    TORCH_CHECK(color.size(2) == 3, "2D color must be [H, W, 3]");
    TORCH_CHECK(distance.size(0) == color.size(0) &&
                    distance.size(1) == color.size(1),
                "distance and color spatial shapes must match");
  } else {
    TORCH_CHECK(color.dim() == 4 && color.size(3) == 3,
                "3D color must be [H, W, nz, 3]");
    TORCH_CHECK(distance.size(0) == color.size(0) &&
                    distance.size(1) == color.size(1) &&
                    distance.size(2) == color.size(2),
                "distance and color spatial shapes must match");
    color_nz = static_cast<int>(color.size(2));
  }

  auto distance_c = distance.contiguous();
  auto color_c = color.contiguous();

  const int height = static_cast<int>(distance_c.size(0));
  const int width = static_cast<int>(distance_c.size(1));
  const int nz = static_cast<int>(distance_c.size(2));

  auto opts = torch::TensorOptions()
                  .dtype(torch::kFloat32)
                  .device(distance_c.device());
  auto elev = torch::empty({height, width}, opts);
  auto cost = torch::empty({height, width}, opts);

  esdf_bev::launch_esdf_to_bev(
      distance_c.data_ptr<float>(), color_c.data_ptr<uint8_t>(),
      elev.data_ptr<float>(), cost.data_ptr<float>(), height, width, nz,
      color_nz, voxel_z, z_min);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              std::string("esdf_to_bev kernel launch failed: ") +
                  cudaGetErrorString(err));
  err = cudaDeviceSynchronize();
  TORCH_CHECK(err == cudaSuccess, std::string("esdf_to_bev kernel failed: ") +
                                      cudaGetErrorString(err));

  return {elev, cost};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("esdf_to_bev", &esdf_to_bev_cuda,
        "Flatten a dense colored ESDF into BEV elevation and cost maps. "
        "color may be [H,W,nz,3] or column-constant [H,W,3].",
        py::arg("distance"), py::arg("color"), py::arg("voxel_z"),
        py::arg("z_min"));
}
