#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel using 2D grid indexing and grid-stride loop
template <typename scalar_t>
__global__ void sigmoid_kernel_2d(const scalar_t* __restrict__ input,
                                    scalar_t* __restrict__ output,
                                    const int64_t size) {
  // Compute a unique block index from the 2D grid layout
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int i = blockId * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x * gridDim.y;

  for (; i < size; i += stride) {
    float x = static_cast<float>(input[i]);
    float res = 1.0f / (1.0f + expf(-x));
    output[i] = static_cast<scalar_t>(res);
  }
}

// Forward function
torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  // Set number of threads per block
  int threads = 256;
  // Total blocks required
  int total_blocks = (size + threads - 1) / threads;

  // Configure a 2D grid to potentially improve workload distribution
  int grid_x = static_cast<int>(sqrtf(static_cast<float>(total_blocks)));
  if (grid_x < 1) grid_x = 1;
  int grid_y = (total_blocks + grid_x - 1) / grid_x;
  dim3 blocks(grid_x, grid_y);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel_2d", [&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    sigmoid_kernel_2d<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
  });

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward (CUDA) with 2D grid indexing");
}
