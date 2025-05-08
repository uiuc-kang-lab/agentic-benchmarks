#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized sigmoid kernel combining multiple element processing and read-only cache (__ldg) usage
// Also using __expf for faster exponential computation (with slightly reduced precision)

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t size) {
  // Calculate global thread index
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Total number of threads in the grid
  int stride = blockDim.x * gridDim.x;

  // Process multiple elements per thread to maximize memory coalescing
  for (int i = tid; i < size; i += stride) {
    // Use __ldg to load input from read-only cache, improving memory throughput
    float x = static_cast<float>(__ldg(&input[i]));
    // Use __expf for a fast exponential approximation
    float exp_val = __expf(-x);
    float sigmoid = 1.0f / (1.0f + exp_val);
    output[i] = static_cast<scalar_t>(sigmoid);
  }
}

// Host function invoked from Python. Allocates output tensor, sets kernel launch configuration,
// Dispatches the kernel for the given type.

torch::Tensor forward(torch::Tensor input) {
  // Allocate output tensor similar to input
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  // Define kernel launch configuration
  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;

  // Dispatch to CUDA kernel based on input scalar type
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", ([&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    
    sigmoid_kernel<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized Sigmoid forward (CUDA) combining memory coalescing and __ldg optimization");
}
