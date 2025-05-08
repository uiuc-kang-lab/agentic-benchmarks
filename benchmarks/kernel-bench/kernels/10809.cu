#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel that uses warp-level primitives for small row lengths
// and falls back to sequential scan for longer rows.

// Unrolled loop for performance improvement
template <typename scalar_t>
__global__ void unrolled_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const scalar_t* x_row = x + idx * L;
    const bool* mask_row = mask + idx * L;
    scalar_t* out_row = output + idx * L;

    scalar_t sum = scalar_t(0);
    #pragma unroll 4
    for (int64_t i = 0; i < L; ++i) {
        if (mask_row[i]) {
            sum += x_row[i];
        }
        out_row[i] = sum;
    }
}

// Host function

torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    // Adjust dimension if negative
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute dimensions to bring the target dimension to the last
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim)
            perm.push_back(i);
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    // Reshape into 2D: each row corresponds to a reduction
    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Determine kernel launch configuration
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        unrolled_masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum with Loop Unrolling (CUDA)");
}
