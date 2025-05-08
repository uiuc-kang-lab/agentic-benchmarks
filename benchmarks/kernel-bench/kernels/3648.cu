#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel that distributes workloads evenly by assigning each thread a contiguous chunk of data
template <typename scalar_t>
__global__ void even_chunk_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               size_t numel) {
  // Compute global thread id and total number of threads
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  // Calculate the number of elements each thread should process (ceiling division)
  size_t items_per_thread = (numel + total_threads - 1) / total_threads;

  // Determine the contiguous block of indices this thread will handle
  size_t start = tid * items_per_thread;
  size_t end = start + items_per_thread;
  if (end > numel) end = numel;

  // Process each element in the assigned contiguous chunk
  for (size_t i = start; i < end; i++) {
    scalar_t x = input[i];
    scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
    // Clamp y to the range [0, 1]
    if (y < static_cast<scalar_t>(0))
      y = static_cast<scalar_t>(0);
    else if (y > static_cast<scalar_t>(1))
      y = static_cast<scalar_t>(1);
    output[i] = y;
  }
}

// Host function that dispatches the kernel
torch::Tensor forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  auto output = torch::empty_like(input);
  size_t numel = input.numel();

  // Configure kernel launch parameters
  // Using 1024 threads per block; blocks is computed to cover all elements
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "even_chunk_hardsigmoid_cuda", ([&] {
    even_chunk_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
      input.data_ptr<scalar_t>(),
      output.data_ptr<scalar_t>(),
      numel);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward (CUDA) with even workload chunk distribution");
}
