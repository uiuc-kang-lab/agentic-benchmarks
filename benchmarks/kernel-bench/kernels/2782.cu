#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int threads = 256;

template <typename scalar_t>
__global__ void sigmoid_kernel_shared_memory(const scalar_t* __restrict__ input,
                                             scalar_t* __restrict__ output,
                                             const int64_t size) {
  __shared__ float shared_input[threads];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;

  if (i < size) {
    shared_input[tid] = static_cast<float>(input[i]);
    __syncthreads();

    float val = -shared_input[tid];
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
  const int blocks = (size + threads - 1) / threads;

  // Dispatch to our CUDA kernel.
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel_shared_memory", [&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();

    sigmoid_kernel_shared_memory<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
  });

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward (CUDA)");
}