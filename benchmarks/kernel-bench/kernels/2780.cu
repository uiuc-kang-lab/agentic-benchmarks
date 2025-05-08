#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Sigmoid kernel optimized for memory coalescing
template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t size) {
  // Calculate global thread index.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Process multiple elements per thread for better coalescing
  int stride = blockDim.x * gridDim.x;
  
  for (int i = tid; i < size; i += stride) {
    float val = static_cast<float>(-input[i]);
    float exp_val = expf(val);
    float r = 1.0f / (1.0f + exp_val);
    output[i] = static_cast<scalar_t>(r);
  }
}

torch::Tensor forward(torch::Tensor input) {
  // Allocate output tensor.
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  // Define CUDA kernel launch configuration.
  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;

  // Dispatch to our CUDA kernel with memory coalescing optimization.
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();

    sigmoid_kernel<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
  });

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward (CUDA) optimized for memory coalescing");
}