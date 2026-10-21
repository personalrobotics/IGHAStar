// CUDA batch rollout for Mason/Lynch PushT QS (see pusht_qs.h).
#include <pusht_qs.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace {

__global__ void pusht_qs_kernel(float *states, float *intermediates,
                                const float *actions, bool *valid,
                                float *pen_inc_out, int n_substeps, int rollouts,
                                float max_agent_vel, float support_radius,
                                float workspace_lo, float workspace_hi,
                                float max_penetration_px) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= rollouts)
    return;

  float state_in[pusht_qs::kStateDim];
  float state_out[pusht_qs::kStateDim];
#pragma unroll
  for (int d = 0; d < pusht_qs::kStateDim; ++d)
    state_in[d] = states[k * pusht_qs::kStateDim + d];

  const float ax = actions[k * 2 + 0];
  const float ay = actions[k * 2 + 1];
  float *inter =
      intermediates ? intermediates + k * n_substeps * pusht_qs::kStateDim
                    : nullptr;
  float pen_inc = 0.f;
  pusht_qs::step_from_snapshot(state_in, ax, ay, state_out, n_substeps, inter,
                               max_agent_vel, &pen_inc, support_radius,
                               workspace_lo, workspace_hi);
#pragma unroll
  for (int d = 0; d < pusht_qs::kStateDim; ++d)
    states[k * pusht_qs::kStateDim + d] = state_out[d];

  bool in_ws = true;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    if (state_out[d] < workspace_lo || state_out[d] > workspace_hi)
      in_ws = false;
  }
  valid[k] = in_ws && (pen_inc <= max_penetration_px);
  if (pen_inc_out)
    pen_inc_out[k] = pen_inc;
}

} // namespace

extern "C" void pusht_qs_launcher(float *h_states, float *h_intermediates,
                                  const float *h_actions, bool *h_valid,
                                  int n_substeps, int rollouts,
                                  float max_agent_vel, float support_radius,
                                  float workspace_lo, float workspace_hi,
                                  float max_penetration_px, float *d_states,
                                  float *d_intermediates, float *d_actions,
                                  bool *d_valid, int blocks, int threads,
                                  cudaStream_t stream) {
  const size_t state_bytes =
      sizeof(float) * size_t(rollouts) * pusht_qs::kStateDim;
  const size_t action_bytes = sizeof(float) * size_t(rollouts) * 2;
  const size_t inter_bytes =
      sizeof(float) * size_t(rollouts) * size_t(n_substeps) * pusht_qs::kStateDim;

  cudaMemcpyAsync(d_states, h_states, state_bytes, cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(d_actions, h_actions, action_bytes, cudaMemcpyHostToDevice,
                  stream);

  pusht_qs_kernel<<<blocks, threads, 0, stream>>>(
      d_states, d_intermediates, d_actions, d_valid, /*pen_inc_out=*/nullptr,
      n_substeps, rollouts, max_agent_vel, support_radius, workspace_lo,
      workspace_hi, max_penetration_px);

  cudaMemcpyAsync(h_states, d_states, state_bytes, cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(h_valid, d_valid, sizeof(bool) * size_t(rollouts),
                  cudaMemcpyDeviceToHost, stream);
  if (h_intermediates && d_intermediates) {
    cudaMemcpyAsync(h_intermediates, d_intermediates, inter_bytes,
                    cudaMemcpyDeviceToHost, stream);
  }
  cudaStreamSynchronize(stream);
}
