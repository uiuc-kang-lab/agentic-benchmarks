#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using atomic operations
template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  float r = 0.0f;

  // Using atomicAdd to safely reduce values in shared memory or global memory if necessary
  if (i < size) {
    float val = static_cast<float>(-input[i]);
    float exp_val = expf(val);
    r = 1.0f / (1.0f + exp_val);
    atomicExch(&output[i], static_cast<scalar_t>(r));
  }
}

torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();

    sigmoid_kernel<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
  });

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward (CUDA) with atomic operations");
}