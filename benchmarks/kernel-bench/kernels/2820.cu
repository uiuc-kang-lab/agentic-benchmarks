#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using grid-stride loop for even workload distribution
template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < size; i += stride) {
    float val = static_cast<float>(-input[i]);
    float exp_val = expf(val);
    float result = 1.0f / (1.0f + exp_val);
    output[i] = static_cast<scalar_t>(result);
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

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", ([&] {
    sigmoid_kernel<scalar_t><<<blocks, threads>>>(input.data_ptr<scalar_t>(),
                                                    output.data_ptr<scalar_t>(),
                                                    size);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Efficient Sigmoid forward (CUDA)");
}
