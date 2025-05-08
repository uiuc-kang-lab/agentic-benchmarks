#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>

// Define number of threads per block
const int THREADS = 256;

// Kernel that computes element-wise sigmoid and performs an intra-block reduction
// using warp-level primitives (__shfl_down_sync) and shared memory to compute the sum
// of sigmoid values for each block.

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 float* block_sums,
                                 const int64_t size) {
  float local_sum = 0.0f;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  // Grid-stride loop to cover all elements
  for (int i = idx; i < size; i += stride) {
    float val = static_cast<float>(input[i]);
    float s = 1.0f / (1.0f + expf(-val));
    output[i] = static_cast<scalar_t>(s);
    local_sum += s;
  }
  
  // Intra-warp reduction using warp shuffle
  unsigned mask = 0xffffffff;
  float warp_sum = local_sum;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    warp_sum += __shfl_down_sync(mask, warp_sum, offset);
  }
  
  // Each warp's lane 0 stores the reduced sum in shared memory
  __shared__ float warp_sums[THREADS / 32];
  int lane = threadIdx.x & 31;  // threadIdx.x % warpSize
  int warpId = threadIdx.x >> 5;  // threadIdx.x / warpSize
  if (lane == 0) {
    warp_sums[warpId] = warp_sum;
  }
  __syncthreads();
  
  // Thread 0 performs final reduction over warp sums and writes block result
  if (threadIdx.x == 0) {
    float block_sum = 0.0f;
    int num_warps = blockDim.x / 32;
    for (int i = 0; i < num_warps; i++) {
      block_sum += warp_sums[i];
    }
    block_sums[blockIdx.x] = block_sum;
  }
}

// Forward function that launches the kernel and then reduces block sums on CPU
// Returns a tuple: (elementwise sigmoid output, total sum of sigmoid values)

std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();
  
  // Determine number of blocks based on number of threads
  int blocks = (size + THREADS - 1) / THREADS;
  
  // Allocate temporary tensor for block-level sums (on CUDA)
  auto block_sums = torch::empty({blocks}, torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));
  
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", ([&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    float* block_sums_data = block_sums.data_ptr<float>();
    sigmoid_kernel<scalar_t><<<blocks, THREADS>>>(input_data, output_data, block_sums_data, size);
  }));
  
  // Reduce block sums on CPU (number of blocks is usually small)
  auto block_sums_cpu = block_sums.cpu();
  float total_sum = 0.0f;
  auto block_ptr = block_sums_cpu.data_ptr<float>();
  for (int i = 0; i < blocks; i++) {
    total_sum += block_ptr[i];
  }
  
  // Create a scalar tensor for the total sum
  auto total_sum_tensor = torch::full({}, total_sum, torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));
  
  return std::make_tuple(output, total_sum_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward with reduction (CUDA)");
}
