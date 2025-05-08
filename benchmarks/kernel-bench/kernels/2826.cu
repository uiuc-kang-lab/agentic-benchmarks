#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sigmoid_kernel_optimized(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          const int64_t size) {
  extern __shared__ float shared_data[];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;

  // Load input into shared memory
  shared_data[tid] = static_cast<float>(-input[i]);
  __syncthreads();

  // Compute sigmoid in shared memory
  float exp_val = expf(shared_data[tid]);
  float r = 1.0f / (1.0f + exp_val);
  output[i] = static_cast<scalar_t>(r);
}

torch::Tensor forward(torch::Tensor input) {
  // Allocate output tensor.
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  // Define CUDA kernel launch configuration.
  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;

  // Dispatch to our CUDA kernel.
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel_optimized", [&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();

    sigmoid_kernel_optimized<scalar_t><<<blocks, threads, threads * sizeof(float)>>>(input_data, output_data, size);
  });

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward optimized (CUDA)");
}