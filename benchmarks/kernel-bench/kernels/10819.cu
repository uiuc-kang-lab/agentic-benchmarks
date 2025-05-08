#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Constant memory configuration
__constant__ int64_t d_N;
__constant__ int64_t d_L;
__constant__ int d_warp_size = 32;

// Optimized device function for sequential processing
template <typename scalar_t>
__device__ __forceinline__ void sequential_masked_cumsum(
    const scalar_t* __restrict__ x_row,
    const bool* __restrict__ mask_row,
    scalar_t* __restrict__ output_row,
    int64_t L) {
    scalar_t sum = 0;
    #pragma unroll 4
    for (int64_t i = 0; i < L; ++i) {
        if (mask_row[i]) {
            sum += x_row[i];
        }
        output_row[i] = sum;
    }
}

template <typename scalar_t>
__global__ void adaptive_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output) {
    
    int row = blockIdx.x;
    if (row >= d_N) return;

    const scalar_t* x_row = x + row * d_L;
    const bool* mask_row = mask + row * d_L;
    scalar_t* out_row = output + row * d_L;

    // For small sequences: parallel processing with warp-level optimization
    if (d_L <= 256) {
        int tid = threadIdx.x;
        int lane = tid & (d_warp_size - 1);
        int warpId = tid >> 5;

        // Shared memory for partial sums
        __shared__ scalar_t warpSums[8];  // Support up to 8 warps
        
        scalar_t val = 0;
        if (tid < d_L) {
            val = mask_row[tid] ? x_row[tid] : 0;
        }

        // Warp-level parallel scan
        #pragma unroll
        for (int offset = 1; offset < d_warp_size; offset <<= 1) {
            scalar_t y = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) val += y;
        }

        if (lane == (d_warp_size - 1) || tid == d_L - 1) {
            warpSums[warpId] = val;
        }
        __syncthreads();

        // Cross-warp accumulation
        if (warpId > 0) {
            scalar_t warpOffset = 0;
            #pragma unroll
            for (int i = 0; i < 8 && i < warpId; i++) {
                warpOffset += warpSums[i];
            }
            val += warpOffset;
        }

        if (tid < d_L) {
            out_row[tid] = val;
        }
    }
    // For larger sequences: optimized sequential processing
    else {
        if (threadIdx.x == 0) {
            sequential_masked_cumsum(x_row, mask_row, out_row, d_L);
        }
    }
}

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

    if (dim < 0) dim += x.dim();
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) perm.push_back(i);
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    int64_t L = x_permuted.size(-1);
    int64_t N = x_permuted.numel() / L;

    cudaMemcpyToSymbol(d_N, &N, sizeof(int64_t));
    cudaMemcpyToSymbol(d_L, &L, sizeof(int64_t));

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    dim3 block(L <= 256 ? 256 : 1);
    dim3 grid(N);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        adaptive_masked_cumsum_kernel<scalar_t><<<grid, block>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>()
        );
    }));

    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    return output_permuted.permute(inv_perm);
}