#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function that computes the cumulative product for a single sequence
// starting at a given base pointer. This modular design improves readability,
// reusability, and maintainability, while preserving the correct sequential logic.

template <typename scalar_t>
__device__ void compute_single_cumprod(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t stride) {
  scalar_t product = 1;
  // Compute the cumulative product along the dimension
  for (int i = 0; i < dim_size; i++) {
    // Access the i-th element along the dim via stride
    int64_t pos = i * stride;
    product *= input[pos];
    output[pos] = product;
  }
}

// Kernel function that uses the modular device function
// Each thread processes one cumprod computation for a given subarray. The grid-stride
// loop ensures that if there are more subarrays than threads, each thread handles
// multiple ones.

template <typename scalar_t>
__global__ void modular_cumprod_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t total_batches,  // total number of subarrays
    const int64_t dim_size,
    const int64_t stride) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global thread index
  int grid_size = gridDim.x * blockDim.x;
  
  // Loop over the subarrays using grid-stride loop
  for (int idx = thread_id; idx < total_batches; idx += grid_size) {
    // Derive the corresponding base offset:
    // Each subarray is identified by a unique combination of indices outside the cumprod dimension.
    // We compute these as:
    //   batch = idx / stride
    //   inner = idx % stride
    // The base_offset into the flattened array is then:
    //   base_offset = batch * (stride * dim_size) + inner
    int batch = idx / stride;
    int inner = idx % stride;
    int64_t base_offset = batch * (stride * dim_size) + inner;
    
    // Call the modular device function for cumulative product on this subarray
    compute_single_cumprod(&input[base_offset], &output[base_offset], dim_size, stride);
  }
}

// Torch binding function

torch::Tensor modular_cumprod_forward(torch::Tensor input, int64_t dim) {
  auto output = torch::empty_like(input);
  
  // Get tensor dimensions and strides
  auto sizes = input.sizes();
  auto strides = input.strides();
  
  // The size along the cumulative product dimension
  int64_t dim_size = sizes[dim];
  // The stride corresponding to the dimension along which we perform cumprod
  int64_t stride = strides[dim];
  // Total number of subarrays that require a cumulative product
  int64_t total_batches = input.numel() / dim_size;
  
  // CUDA kernel launch parameters
  const int threads = 512;
  const int blocks = (total_batches + threads - 1) / threads;
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "modular_cumprod_forward", ([&] {
    modular_cumprod_kernel<scalar_t><<<blocks, threads>>>(
      output.data_ptr<scalar_t>(),
      input.data_ptr<scalar_t>(),
      total_batches,
      dim_size,
      stride
    );
  }));
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &modular_cumprod_forward, "Modular cumulative product forward (CUDA)");
}
