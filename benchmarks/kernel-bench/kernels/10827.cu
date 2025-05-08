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

// Warp-level inclusive scan with shfl_down for small segments

template <typename scalar_t>
__device__ scalar_t warp_inclusive_scan(scalar_t val, const unsigned mask) {
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        scalar_t y = __shfl_down_sync(mask, val, offset);
        if (lane + offset < WARP_SIZE) val += y;
    }
    return val;
}


// Hybrid kernel: leveraging warp-intrinsics as recommended for small reductions
// For small L (<= PARALLEL_THRESHOLD) use warp-specialized scan
// For larger L, use an optimized tiled shared-memory scan

template <typename scalar_t>
__global__ void warp_specialized_masked_cumsum_kernel(
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
    unsigned mask_active = __ballot_sync(0xffffffff, true);

    // Use warp-specialized scan for small rows
    if (L <= PARALLEL_THRESHOLD) {
        scalar_t val = 0;
        if (tid < L) {
            val = apply_mask(x_row[tid], mask_row[tid]);
        }
        scalar_t scan_val = warp_inclusive_scan(val, mask_active);

        if (tid < L) {
            out_row[tid] = scan_val;
        }
    } else {

        // Tiled scan for larger rows
        __shared__ scalar_t tile_data[TILE_SIZE];
        __shared__ scalar_t row_offset;
        if (tid == 0) row_offset = 0;
        __syncthreads();

        int num_tiles = (L + TILE_SIZE - 1) / TILE_SIZE;
        for (int t = 0; t < num_tiles; t++) {
            int tile_start = t * TILE_SIZE;
            int idx = tile_start + tid;
            int count = min(TILE_SIZE, L - tile_start);

            scalar_t val = 0;
            if (tid < count) {
                val = apply_mask(x_row[idx], mask_row[idx]);
            }
            tile_data[tid] = val;
            __syncthreads();

            // Parallel scan within tile using shared memory
            for (int offset = 1; offset < count; offset *= 2) {
                scalar_t temp = 0;
                if (tid >= offset && tid < count) {
                    temp = tile_data[tid - offset];
                }
                __syncthreads();
                if (tid < count) {
                    tile_data[tid] += temp;
                }
                __syncthreads();
            }

            // Write results with offset
            if (tid < count) {
                out_row[idx] = tile_data[tid] + row_offset;
            }
            
            // Update running offset
            if (tid == count - 1) {
                row_offset += tile_data[tid];
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
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "warp_specialized_masked_cumsum", ([&] {
        warp_specialized_masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &masked_cumsum, "Warp Specialized Masked Cumulative Sum (CUDA)");
}
