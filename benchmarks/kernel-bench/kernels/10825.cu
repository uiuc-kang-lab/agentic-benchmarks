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

template <typename scalar_t>
__device__ __forceinline__ void warp_scan(
    scalar_t& val,
    scalar_t* warp_sums,
    const int lane,
    const int warp_id
) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        scalar_t y = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += y;
    }
    
    if (lane == WARP_SIZE-1) {
        warp_sums[warp_id] = val;
    }
}

template <typename scalar_t>
__device__ __forceinline__ void process_small_row(
    const scalar_t* __restrict__ x_row,
    const bool* __restrict__ mask_row,
    scalar_t* __restrict__ out_row,
    const int tid,
    const int L
) {
    const int lane = tid & (WARP_SIZE-1);
    const int warp_id = tid >> 5;
    
    scalar_t val = 0;
    if (tid < L) {
        val = apply_mask(x_row[tid], mask_row[tid]);
    }
    
    __shared__ scalar_t warp_sums[WARP_SIZE];  // Support up to 32 warps
    
    warp_scan(val, warp_sums, lane, warp_id);
    __syncthreads();
    
    if (warp_id > 0) {
        scalar_t warp_offset = 0;
        if (lane == 0) {
            for (int i = 0; i < warp_id; i++) {
                warp_offset += warp_sums[i];
            }
        }
        warp_offset = __shfl_sync(0xffffffff, warp_offset, 0);
        val += warp_offset;
    }
    
    if (tid < L) {
        out_row[tid] = val;
    }
}

template <typename scalar_t>
__device__ __forceinline__ void process_tile(
    const scalar_t* __restrict__ x_row,
    const bool* __restrict__ mask_row,
    scalar_t* __restrict__ out_row,
    scalar_t& row_offset,
    const int tile_idx,
    const int L
) {
    __shared__ scalar_t tile_data[TILE_SIZE];
    const int tid = threadIdx.x;
    const int tile_start = tile_idx * TILE_SIZE;
    const int idx = tile_start + tid;
    const int count = min(TILE_SIZE, L - tile_start);
    
    // Load and mask data
    scalar_t val = 0;
    if (tid < count) {
        val = apply_mask(x_row[idx], mask_row[idx]);
    }
    tile_data[tid] = val;
    __syncthreads();
    
    // Parallel scan within tile
    #pragma unroll
    for (int offset = 1; offset < TILE_SIZE; offset *= 2) {
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
    
    // Write result with offset
    if (tid < count) {
        out_row[idx] = tile_data[tid] + row_offset;
    }
    
    // Update running offset
    if (tid == count-1) {
        row_offset += tile_data[tid];
    }
    __syncthreads();
}

template <typename scalar_t>
__global__ void modular_hybrid_masked_cumsum_kernel(
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
    
    if (L <= PARALLEL_THRESHOLD) {
        process_small_row(x_row, mask_row, out_row, threadIdx.x, L);
    } else {
        __shared__ scalar_t row_offset;
        if (threadIdx.x == 0) row_offset = 0;
        __syncthreads();
        
        const int num_tiles = (L + TILE_SIZE - 1) / TILE_SIZE;
        for (int tile = 0; tile < num_tiles; tile++) {
            process_tile(x_row, mask_row, out_row, row_offset, tile, L);
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
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "modular_hybrid_masked_cumsum", ([&] {
        modular_hybrid_masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &masked_cumsum, "Modular Hybrid Masked Cumulative Sum (CUDA)");
}