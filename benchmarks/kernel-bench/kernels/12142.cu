#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel for computing hinge loss
__global__ void hinge_loss_kernel(const float* __restrict__ predictions, 
                                    const float* __restrict__ targets, 
                                    float* __restrict__ output, 
                                    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    float prod = predictions[i] * targets[i];
    output[i] = (prod < 1.0f) ? (1.0f - prod) : 0.0f;
  }
}

// Forward function with an optional block size parameter
// If block_size_opt is not provided (or <= 0), dynamic occupancy tuning is used
// and then clamped to one of the candidate block sizes {32, 64, 128, 256, 512}.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets, int block_size_opt = -1) {
  CHECK_INPUT(predictions);
  CHECK_INPUT(targets);

  int n = predictions.numel();
  torch::Tensor output = torch::empty_like(predictions);

  // If no block size is specified, use cudaOccupancyMaxPotentialBlockSize to auto-tune
  if (block_size_opt <= 0) {
    int auto_block_size = 0;
    int minGridSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &auto_block_size, hinge_loss_kernel, 0, n);
    block_size_opt = auto_block_size;
  }

  // Clamp the chosen block size to one of the candidate values: 32, 64, 128, 256, 512
  const int candidates[5] = {32, 64, 128, 256, 512};
  int best_candidate = candidates[0];
  int best_diff = abs(block_size_opt - candidates[0]);
  for (int i = 1; i < 5; i++) {
    int diff = abs(block_size_opt - candidates[i]);
    if (diff < best_diff) {
      best_diff = diff;
      best_candidate = candidates[i];
    }
  }
  block_size_opt = best_candidate;

  int blocks = (n + block_size_opt - 1) / block_size_opt;
  blocks = std::min(blocks, 65535);

  hinge_loss_kernel<<<blocks, block_size_opt>>>(
      predictions.data_ptr<float>(),
      targets.data_ptr<float>(),
      output.data_ptr<float>(),
      n);

  return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Hinge Loss Forward with Tunable Block Size",
        py::arg("predictions"), py::arg("targets"), py::arg("block_size_opt") = -1);
}
