#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel uses grid-stride loops to ensure that all rows are processed even if the total
// number of rows exceeds the number of available threads. Each thread processes one or more rows
// sequentially, handling boundaries correctly.

template <typename scalar_t>
__global__ void masked_cumsum_kernel_stride(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; row < N; row += stride) {
    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* output_row = output + row * L;
    scalar_t sum = static_cast<scalar_t>(0);
    for (int64_t i = 0; i < L; i++) {
      if (mask_row[i]) {
        sum += x_row[i];
      }
      output_row[i] = sum;
    }
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {
  CHECK_INPUT(x);
  CHECK_INPUT(mask);
  TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

  if (dim < 0) {
    dim += x.dim();
  }
  TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

  // Permute dimensions to bring the target dimension (for cumulative sum) to the last
  std::vector<int64_t> perm;
  for (int64_t i = 0; i < x.dim(); i++) {
    if (i != dim) {
      perm.push_back(i);
    }
  }
  perm.push_back(dim);

  auto x_perm = x.permute(perm).contiguous();
  auto mask_perm = mask.permute(perm).contiguous();

  int64_t N = x_perm.numel() / x_perm.size(-1);
  int64_t L = x_perm.size(-1);

  auto x_flat = x_perm.view({N, L});
  auto mask_flat = mask_perm.view({N, L});
  auto output_flat = torch::empty_like(x_flat);

  const int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_kernel_stride", ([&] {
    masked_cumsum_kernel_stride<scalar_t><<<gridSize, blockSize>>>(
        x_flat.data_ptr<scalar_t>(),
        mask_flat.data_ptr<bool>(),
        output_flat.data_ptr<scalar_t>(),
        N, L);
  }));

  auto output_perm = output_flat.view(x_perm.sizes());
  std::vector<int64_t> inv_perm(perm.size());
  for (size_t i = 0; i < perm.size(); i++) {
    inv_perm[perm[i]] = i;
  }
  auto output = output_perm.permute(inv_perm);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &masked_cumsum, "Masked Cumulative Sum with Stride Loops (CUDA)");
}
