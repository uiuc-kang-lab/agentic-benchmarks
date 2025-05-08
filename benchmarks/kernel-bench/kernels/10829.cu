#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define PARALLEL_THRESHOLD 256
#define WARP_SIZE 32
#define TILE_SIZE 256

template <typename scalar_t>
__device__ __forceinline__ scalar_t apply_mask(scalar_t val, bool mask) {
    return mask ? val : static_cast<scalar_t>(0);
}

// Warp-level inclusive scan using shfl_down_sync
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_inclusive_scan(scalar_t val, const int lane) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        scalar_t y = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += y;
    }
    return val;
}

// Hybrid kernel with warp-level primitives for optimization
template <typename scalar_t>
__global__ void warp_optimized_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    const int64_t N,
    const int64_t L
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    
    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* out_row = output + row * L;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;

    if (L <= PARALLEL_THRESHOLD) {
        scalar_t val = 0;
        if (tid < L) {
            val = apply_mask(x_row[tid], mask_row[tid]);
        }
        
        val = warp_inclusive_scan(val, lane);

        __shared__ scalar_t warp_sums[WARP_SIZE];
        if (lane == WARP_SIZE - 1 || tid == L - 1) {
            warp_sums[warpId] = val;
        }
        __syncthreads();

        scalar_t warp_offset = 0;
        if (warpId > 0) {
            if (lane == 0) {
                for (int i = 0; i < warpId; i++) {
                    warp_offset += warp_sums[i];
                }
            }
            warp_offset = __shfl_sync(0xffffffff, warp_offset, 0);
            val += warp_offset;
        }

        if (tid < L) {
            out_row[tid] = val;
        }
    } else {
        __shared__ scalar_t row_offset;
        if (tid == 0) row_offset = 0;
        __syncthreads();

        int num_tiles = (L + TILE_SIZE - 1) / TILE_SIZE;
        for (int tile = 0; tile < num_tiles; tile++) {
            int tile_start = tile * TILE_SIZE;
            int idx = tile_start + tid;
            int count = min(TILE_SIZE, L - tile_start);

            scalar_t val = 0;
            if (tid < count) {
                val = apply_mask(x_row[idx], mask_row[idx]);
            }
            
            // Inclusive scan within tile
            for (int offset = 1; offset < TILE_SIZE; offset *= 2) {
                scalar_t y = __shfl_up_sync(0xffffffff, val, offset);
                if (tid >= offset) val += y;
            }
            
            if (tid < count) {
                out_row[idx] = val + row_offset;
            }

            if (tid == count - 1) {
                atomicAdd(&row_offset, val);
            }
            __syncthreads();
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
    
    const int64_t L = x_permuted.size(-1);
    const int64_t N = x_permuted.numel() / L;
    
    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);
    
    const int threads = TILE_SIZE;
    const int blocks = N;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "warp_optimized_masked_cumsum", ([&] {
        warp_optimized_masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &masked_cumsum, "Warp Optimized Masked Cumulative Sum (CUDA)");
}