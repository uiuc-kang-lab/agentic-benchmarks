#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel using grid-stride loops to handle workloads larger than available threads
// Each thread processes multiple rows by iterating over them using a stride loop.
template <typename scalar_t>
__global__ void stride_loop_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {
  // Calculate a global thread index and stride
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Loop over rows with a grid-stride loop
  for (int row = index; row < N; row += stride) {
    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* out_row = output + row * L;
    scalar_t sum = static_cast<scalar_t>(0);

    // Compute the masked cumulative sum for this row
    for (int64_t i = 0; i < L; ++i) {
      if (mask_row[i]) {
        sum += x_row[i];
      }
      out_row[i] = sum;
    }
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Host function to set up and launch the CUDA kernel
torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

  CHECK_INPUT(x);
  CHECK_INPUT(mask);
  TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

  // Adjust dimension for negative values
  if (dim < 0) {
    dim += x.dim();
  }
  TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

  // Permutation to bring the operation dimension to the last
  std::vector<int64_t> perm;
  for (int64_t i = 0; i < x.dim(); ++i) {
    if (i != dim)
      perm.push_back(i);
  }
  perm.push_back(dim);

  auto x_permuted = x.permute(perm).contiguous();
  auto mask_permuted = mask.permute(perm).contiguous();

  // Reshape into 2D: N rows and L columns
  int64_t L = x_permuted.size(-1);
  int64_t N = x_permuted.numel() / L;

  auto x_flat = x_permuted.view({N, L});
  auto mask_flat = mask_permuted.view({N, L});
  auto output_flat = torch::empty_like(x_flat);

  // Set up grid-stride loop launch configuration
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
    stride_loop_masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
        x_flat.data_ptr<scalar_t>(),
        mask_flat.data_ptr<bool>(),
        output_flat.data_ptr<scalar_t>(),
        N,
        L
    );
  }));

  // Reshape and permute back to the original shape
  auto output_permuted = output_flat.view(x_permuted.sizes());
  std::vector<int64_t> inv_perm(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    inv_perm[perm[i]] = i;
  }
  auto output = output_permuted.permute(inv_perm);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &masked_cumsum, "Masked Cumulative Sum with Stride Loop (CUDA)");
}
