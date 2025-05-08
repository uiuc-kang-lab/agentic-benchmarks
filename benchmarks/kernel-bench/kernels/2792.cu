/*
 * Optimized Sigmoid Kernel combining shared memory tiling (Kernel 2) with multi-element per thread loop (Kernel 1).
 * This kernel loads a tile of data from global memory into shared memory in a coalesced manner, 
 * computes the sigmoid function on the tile, and writes the result back to global memory.
 * For data sizes larger than one tile per block, a global stride loop processes remaining elements.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int THREADS = 256;
constexpr int ELEMENTS_PER_THREAD = 4;             // Number of elements each thread processes in the tile
constexpr int TILE_SIZE = THREADS * ELEMENTS_PER_THREAD; // Total elements per block (tile size)

// Optimized sigmoid kernel using shared memory tiling and loop unrolling
template <typename scalar_t>
__global__ void optimized_sigmoid_kernel(const scalar_t* __restrict__ input,
                                           scalar_t* __restrict__ output,
                                           const int64_t size) {
  // Calculate the start index of the current tile for this block
  int tile_start = blockIdx.x * TILE_SIZE;

  // Allocate shared memory for the tile
  __shared__ float tile[TILE_SIZE];

  // Load elements from global memory to shared memory in a coalesced manner
  #pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
    int idx = tile_start + threadIdx.x + i * blockDim.x;
    if (idx < size) {
      tile[threadIdx.x + i * blockDim.x] = static_cast<float>(input[idx]);
    }
  }
  __syncthreads();

  // Compute the sigmoid function on the data from shared memory
  #pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
    int idx = tile_start + threadIdx.x + i * blockDim.x;
    if (idx < size) {
      float val = -tile[threadIdx.x + i * blockDim.x];
      float exp_val = expf(val);
      float r = 1.0f / (1.0f + exp_val);
      output[idx] = static_cast<scalar_t>(r);
    }
  }

  // If the data is larger than one tile per block, process remaining elements.
  int stride = gridDim.x * TILE_SIZE;
  for (int base = tile_start + stride; base < size; base += stride) {
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
      int idx = base + threadIdx.x + i * blockDim.x;
      if (idx < size) {
        float value = static_cast<float>(input[idx]);
        float r = 1.0f / (1.0f + expf(-value));
        output[idx] = static_cast<scalar_t>(r);
      }
    }
  }
}

// C++ interface exposed to Python
torch::Tensor forward(torch::Tensor input) {
  // Create an output tensor with the same shape as input
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  // Determine number of blocks needed
  int blocks = (size + TILE_SIZE - 1) / TILE_SIZE;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_sigmoid_kernel", ([&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    optimized_sigmoid_kernel<scalar_t><<<blocks, THREADS>>>(input_data, output_data, size);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized Sigmoid Forward (CUDA) using shared memory tiling and coalesced accesses");
}
