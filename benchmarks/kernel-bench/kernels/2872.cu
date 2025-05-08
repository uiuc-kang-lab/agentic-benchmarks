#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel function to compute sigmoid
template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t size) {
  const int stride = blockDim.x * gridDim.x * blockDim.y;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
    float val = static_cast<float>(-input[i]);
    float exp_val = expf(val);
    float r = 1.0f / (1.0f + exp_val);
    output[i] = static_cast<scalar_t>(r);
  }
}

// Forward function with CUDA streams
torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  const int threads = 256;
  const int max_blocks = 65535;  // Maximum blocks per grid dimension
  const int min_blocks = (size + threads - 1) / threads;
  const int blocks = min(max_blocks, min_blocks);

  // Create CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();

    // Launch kernel on the created stream
    sigmoid_kernel<scalar_t><<<blocks, threads, 0, stream>>>(input_data, output_data, size);
  });

  // Synchronize the stream
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward (CUDA) with streams");
}