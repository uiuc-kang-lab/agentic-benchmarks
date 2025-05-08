#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel that minimizes warp divergence by using arithmetic to replace conditionals
// for the warp-level inclusive scan when L <= 32. For larger L, a branchless sequential loop is used.

template <typename scalar_t>
__global__ void warp_scan_nobranches_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // For rows that fit within a single warp
    if (L <= 32) {
        // Each warp handles one row
        int warpId = idx / 32;
        int lane = idx % 32;
        if (warpId >= N) return;

        const scalar_t* x_row = x + warpId * L;
        const bool* mask_row = mask + warpId * L;
        scalar_t* out_row = output + warpId * L;

        // Load the value in a branch-minimized way
        // We cannot read out-of-bound; only threads with lane < L load real data, others get 0
        scalar_t val = (lane < L) ? ( mask_row[lane] ? x_row[lane] : static_cast<scalar_t>(0) ) : static_cast<scalar_t>(0);

        unsigned int full_mask = 0xffffffff;  // full warp mask
        // Inclusive warp scan using shuffle intrinsics, with branchless update
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            scalar_t n = __shfl_up_sync(full_mask, val, offset);
            // Replace conditional add: use arithmetic flag (1 if lane >= offset, 0 otherwise)
            int flag = (lane >= offset);
            val += n * flag;
        }

        // Write back the result for valid lanes
        if (lane < L) {
            out_row[lane] = val;
        }
    } else {
        // Fallback for rows longer than a warp: each thread processes one row using a branchless loop
        int row = idx;
        if (row >= N) return;

        const scalar_t* x_row = x + row * L;
        const bool* mask_row = mask + row * L;
        scalar_t* out_row = output + row * L;
        scalar_t sum = static_cast<scalar_t>(0);
        
        // Use multiplication by casted boolean to avoid branching
        for (int i = 0; i < L; i++) {
            sum += static_cast<scalar_t>(mask_row[i]) * x_row[i];
            out_row[i] = sum;
        }
    }
}

// Host function to prepare tensors and launch the kernel

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

    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute dimensions to bring the target dimension to the last
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) {
            perm.push_back(i);
        }
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Determine thread configuration
    int total_threads = (L <= 32) ? (N * 32) : N;
    int threadsPerBlock = 256;
    int blocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        warp_scan_nobranches_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum with warp-level scan and minimized divergence (CUDA)");
}
