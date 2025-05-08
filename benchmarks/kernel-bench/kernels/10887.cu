#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel: Use __ldg() for read-only global loads and unroll loop by a factor of 4 to improve throughput.
// Assumes input tensor is contiguous and 128-bit aligned for optimal load performance.

template <typename scalar_t>
__global__ void masked_cumsum_kernel_ldg_aligned(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    // Each block processes one row of length L
    int row = blockIdx.x;
    if (row >= N) return;

    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* output_row = output + row * L;

    scalar_t sum = 0;
    int i = 0;
    // Process in chunks of 4 for better memory throughput with unrolling
    int L4 = (L / 4) * 4;
    for (; i < L4; i += 4) {
        scalar_t a0 = __ldg(x_row + i);
        bool m0 = __ldg(mask_row + i);
        sum += (m0 ? a0 : scalar_t(0));
        output_row[i] = sum;

        scalar_t a1 = __ldg(x_row + i + 1);
        bool m1 = __ldg(mask_row + i + 1);
        sum += (m1 ? a1 : scalar_t(0));
        output_row[i + 1] = sum;

        scalar_t a2 = __ldg(x_row + i + 2);
        bool m2 = __ldg(mask_row + i + 2);
        sum += (m2 ? a2 : scalar_t(0));
        output_row[i + 2] = sum;

        scalar_t a3 = __ldg(x_row + i + 3);
        bool m3 = __ldg(mask_row + i + 3);
        sum += (m3 ? a3 : scalar_t(0));
        output_row[i + 3] = sum;
    }
    // Remainder loop
    for (; i < L; i++) {
        scalar_t a = __ldg(x_row + i);
        bool m = __ldg(mask_row + i);
        sum += (m ? a : scalar_t(0));
        output_row[i] = sum;
    }
}

// Macros for input validation
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward function
torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    CHECK_INPUT(x);
    CHECK_INPUT(mask);
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    // Adjust dimension
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute dimensions to bring the reduction dim to the last dimension
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) {
            perm.push_back(i);
        }
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    // Reshape to 2D: each row is processed independently
    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Launch one block per row, with one thread per block as the computation is sequential
    const int blocks = N;
    const int threads = 1;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda_ldg_aligned", ([&] {
        masked_cumsum_kernel_ldg_aligned<scalar_t><<<blocks, threads>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    // Reshape and permute back to the original layout
    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum with __ldg() optimized global loads (CUDA)");
}
