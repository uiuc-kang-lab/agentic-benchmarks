#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a grid-stride loop with manual unrolling by a factor of 4. Each thread loads
// and computes four sigmoid values per outer loop iteration, reducing loop overhead.

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t size) {
  // Each thread's initial index
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x;
  int i = idx;
  
  // Process groups of four elements at a time
  for (; i + 3 * stride < size; i += stride * 4) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int index = i + j * stride;
      float val = static_cast<float>(-input[index]);
      float exp_val = expf(val);
      float r = 1.0f / (1.0f + exp_val);
      output[index] = static_cast<scalar_t>(r);
    }
  }
  
  // Process any remaining elements
  for (; i < size; i += stride) {
    float val = static_cast<float>(-input[i]);
    float exp_val = expf(val);
    float r = 1.0f / (1.0f + exp_val);
    output[i] = static_cast<scalar_t>(r);
  }
}


torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  // Standard kernel launch configuration
  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;

  // Dispatch over possible floating point types
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", ([&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    sigmoid_kernel<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward (CUDA) with loop unrolling");
}
