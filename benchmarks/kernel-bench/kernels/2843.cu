#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel using grid-stride loop for optimal workload distribution
// and utilizes shared memory for caching input data into the block scope
template <typename scalar_t>
__global__ void efficient_sigmoid_kernel(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         const int64_t size) {
  extern __shared__ float shared_input[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Load input data into shared memory
  if (threadIdx.x < size) {
    shared_input[threadIdx.x] = static_cast<float>(-input[idx]);
  }
  __syncthreads();

  for (int i = idx; i < size; i += stride) {
    if (i < size) {
      float val = shared_input[threadIdx.x];
      float exp_val = expf(val);
      float r = 1.0f / (1.0f + exp_val);
      output[i] = static_cast<scalar_t>(r);
    }
    __syncthreads();
  }
}

// Forward function launching the kernel
torch::Tensor forward(torch::Tensor input) {
  // Allocate output tensor
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  // Configure kernel launch parameters
  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;

  // Define the shared memory size 
  const int shared_memory_size = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "efficient_sigmoid_kernel", ([&] {
    efficient_sigmoid_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(input.data_ptr<scalar_t>(),
                                                                                output.data_ptr<scalar_t>(),
                                                                                size);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Efficient Sigmoid forward (CUDA with shared memory)");
}
