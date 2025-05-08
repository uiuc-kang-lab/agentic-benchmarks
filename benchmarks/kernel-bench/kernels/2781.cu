#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel leverages shared memory to load a block of input values,
// compute the sigmoid function on them, and write back the result to global memory.
// Each thread loads its corresponding element into shared memory, ensuring
// that global memory accesses are minimized per block.


template <typename scalar_t>
__global__ void sigmoid_kernel_shared(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        const int64_t size) {
  extern __shared__ scalar_t shared_data[];  // dynamically-allocated shared memory
  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;

  // Load from global memory into shared memory
  if (index < size) {
    shared_data[tid] = input[index];
  }
  __syncthreads();

  // Compute sigmoid using the shared memory value
  if (index < size) {
    float val = -static_cast<float>(shared_data[tid]);
    float exp_val = expf(val);
    float r = 1.0f / (1.0f + exp_val);
    output[index] = static_cast<scalar_t>(r);
  }
}


// Forward function that prepares data and launches the kernel
torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();

  // Configure the CUDA kernel launch configuration
  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel_shared", ([&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    // Allocate shared memory: one element per thread
    sigmoid_kernel_shared<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
        input_data, output_data, size);
  }));

  return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward with shared memory (CUDA)");
}
