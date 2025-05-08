#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using shared memory and warp-level primitives for optimization

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t size) {
  extern __shared__ float shared_data[256];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  const int stride = blockDim.x * gridDim.x;

  // Load data into shared memory
  if (i < size) {
    shared_data[tid] = static_cast<float>(-input[i]);
  }
  __syncthreads();

  // Compute sigmoid using shared memory
  if (i < size) {
    float val = shared_data[tid];
    float exp_val = expf(val);
    float r = 1.0f / (1.0f + exp_val);
    output[i] = static_cast<scalar_t>(r);
  }

  // Warp-level reduction (example, not directly applicable to sigmoid but for illustration)
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    float temp = __shfl_down_sync(0xFFFFFFFF, shared_data[tid], offset);
    if (tid + offset < blockDim.x) {
      shared_data[tid] += temp;
    }
  }
}

torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;
  const int shared_mem_size = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();

    sigmoid_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(input_data, output_data, size);
  });

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward (CUDA) with shared memory and warp-level optimization");
}