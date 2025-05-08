#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>

// Define the maximum number of elements allowed in constant memory.
// For safety on typical hardware (e.g., 64KB total constant memory), we use 4096 elements.
// This implies that the product N*L must not exceed 4096.
#define CONST_MEM_SIZE 4096

// Declare constant memory arrays for the input x data and mask.
// We provide separate arrays for float and double data types.
__constant__ float d_x_const_float[CONST_MEM_SIZE];
__constant__ double d_x_const_double[CONST_MEM_SIZE];
__constant__ bool d_mask_const[CONST_MEM_SIZE];

// CUDA kernel for masked cumulative sum for float data using constant memory
__global__ void masked_cumsum_kernel_const_float(
    float* __restrict__ output,
    int64_t N,
    int64_t L) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  int row_offset = idx * L;
  float sum = 0.0f;
  for (int64_t i = 0; i < L; ++i) {
    bool m_val = d_mask_const[row_offset + i];
    float x_val = d_x_const_float[row_offset + i];
    if (m_val) {
      sum += x_val;
    }
    output[row_offset + i] = sum;
  }
}

// CUDA kernel for masked cumulative sum for double data using constant memory
__global__ void masked_cumsum_kernel_const_double(
    double* __restrict__ output,
    int64_t N,
    int64_t L) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  int row_offset = idx * L;
  double sum = 0.0;
  for (int64_t i = 0; i < L; ++i) {
    bool m_val = d_mask_const[row_offset + i];
    double x_val = d_x_const_double[row_offset + i];
    if (m_val) {
      sum += x_val;
    }
    output[row_offset + i] = sum;
  }
}

// Host function implementing masked cumulative sum using constant memory.
// The read-only inputs "x" and "mask" are copied into constant memory before launching the kernel.
// This optimization takes advantage of the low latency and broadcast capabilities of constant memory
// provided that the flattened tensor size (N*L) does not exceed CONST_MEM_SIZE.

torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

  // Basic checks
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
  TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

  // Adjust dim to be non-negative
  if (dim < 0) {
    dim += x.dim();
  }
  TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

  // Permute dimensions to bring the target dim to the last
  std::vector<int64_t> perm;
  for (int64_t i = 0; i < x.dim(); ++i) {
    if (i != dim)
      perm.push_back(i);
  }
  perm.push_back(dim);

  auto x_permuted = x.permute(perm).contiguous();
  auto mask_permuted = mask.permute(perm).contiguous();

  // Reshape to 2D: [N, L]
  int64_t N = x_permuted.numel() / x_permuted.size(-1);
  int64_t L = x_permuted.size(-1);

  auto x_flat = x_permuted.view({N, L});
  auto mask_flat = mask_permuted.view({N, L});
  auto output_flat = torch::empty_like(x_flat);

  // Ensure that the total number of elements fits within our constant memory limit
  TORCH_CHECK(x_flat.numel() <= CONST_MEM_SIZE, "Tensor size exceeds constant memory limits");

  // Copy the read-only data into constant memory.
  // Since the tensor is already on the GPU, use cudaMemcpyDeviceToDevice.
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda_const", ([&] {
    if (std::is_same<scalar_t, float>::value) {
      cudaMemcpyToSymbol(
          d_x_const_float,
          x_flat.data_ptr<scalar_t>(),
          sizeof(scalar_t) * x_flat.numel(),
          0,
          cudaMemcpyDeviceToDevice);
    } else if (std::is_same<scalar_t, double>::value) {
      cudaMemcpyToSymbol(
          d_x_const_double,
          x_flat.data_ptr<scalar_t>(),
          sizeof(scalar_t) * x_flat.numel(),
          0,
          cudaMemcpyDeviceToDevice);
    }
  }));

  cudaMemcpyToSymbol(
      d_mask_const,
      mask_flat.data_ptr<bool>(),
      sizeof(bool) * mask_flat.numel(),
      0,
      cudaMemcpyDeviceToDevice);

  // Launch the kernel with one thread per row
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda_const", ([&] {
    if (std::is_same<scalar_t, float>::value) {
      masked_cumsum_kernel_const_float<<<blocks, threads>>>(
          output_flat.data_ptr<scalar_t>(),
          N,
          L);
    } else if (std::is_same<scalar_t, double>::value) {
      masked_cumsum_kernel_const_double<<<blocks, threads>>>(
          output_flat.data_ptr<scalar_t>(),
          N,
          L);
    }
  }));

  // Reshape and permute the output back to the original tensor order
  auto output_permuted = output_flat.view(x_permuted.sizes());
  std::vector<int64_t> inv_perm(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    inv_perm[perm[i]] = i;
  }
  auto output = output_permuted.permute(inv_perm);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &masked_cumsum, "Masked Cumulative Sum with constant memory (CUDA)");
}
