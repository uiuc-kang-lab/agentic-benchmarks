/*
Hybrid Masked Cumulative Sum CUDA Kernel

This kernel combines two strategies for computing a masked cumulative sum along a given dimension of an input tensor:

1. For rows with a small number of elements (L <= PARALLEL_THRESHOLD), we use a warp-level scan with shuffle intrinsics. This minimizes synchronization overhead and fully leverages intra-warp communication.

2. For rows with a larger number of elements (L > PARALLEL_THRESHOLD), we partition each row into contiguous tiles (of size equal to the block width, here 256 threads) and perform an in-block parallel scan on each tile using shared memory. The partial sums of each tile are then accumulated via a running offset maintained in shared memory. This tiled approach yields a fully parallel scan over larger rows.

The host code permutes the tensor dimensions to bring the target dimension to the last axis, reshapes the data into a 2D tensor (N rows, L columns), and launches one CUDA block per row. After processing, the output is reshaped and permuted back to the original tensor layout.

Author: Expert CUDA Engineer
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Threshold to decide which scan method to use
#define PARALLEL_THRESHOLD 256

// Utility: apply the mask and return x if mask true, else 0
template <typename scalar_t>
__device__ __forceinline__ scalar_t apply_mask_val(const scalar_t x, const bool m) {
    return m ? x : static_cast<scalar_t>(0);
}

// Hybrid kernel: each block processes one row
// For small L (<= PARALLEL_THRESHOLD) use warp-scan, otherwise use tiled shared-memory scan
template <typename scalar_t>
__global__ void hybrid_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L
) {
    int row = blockIdx.x;
    if (row >= N) return;
    
    // Pointers to the current row
    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* out_row = output + row * L;

    // Case 1: Use warp-level scan when L is small
    if (L <= PARALLEL_THRESHOLD) {
        int tid = threadIdx.x;
        int lane = tid & 31;      // lane index in the warp
        int warpId = tid >> 5;    // warp index within the block
        
        scalar_t val = static_cast<scalar_t>(0);
        if (tid < L) {
            val = apply_mask_val(x_row[tid], mask_row[tid]);
        }
        
        // Inclusive warp-scan using shuffle intrinsics
        for (int offset = 1; offset < 32; offset <<= 1) {
            scalar_t y = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) {
                val += y;
            }
        }

        __shared__ scalar_t warpSums[32]; // one entry per warp (max warps = blockDim.x/32)
        if ((lane == 31) || (tid == L - 1)) {
            warpSums[warpId] = val;
        }
        __syncthreads();

        scalar_t warpOffset = 0;
        if (warpId > 0) {
            // Let thread 0 of each warp accumulate the sum of preceding warps
            if (lane == 0) {
                for (int i = 0; i < warpId; i++) {
                    warpOffset += warpSums[i];
                }
            }
            warpOffset = __shfl_sync(0xffffffff, warpOffset, 0);
            val += warpOffset;
        }

        if (tid < L) {
            out_row[tid] = val;
        }
    } else {
        // Case 2: For larger L, use a tiled scan within each row
        // We assume the block has 256 threads which we use as the tile size.
        const int tile_size = blockDim.x;  // = 256
        __shared__ scalar_t tile_data[256];
        __shared__ scalar_t row_offset;  // running offset across tiles
        if (threadIdx.x == 0) row_offset = 0;
        __syncthreads();

        int num_tiles = (L + tile_size - 1) / tile_size;
        for (int t = 0; t < num_tiles; t++) {
            int tile_start = t * tile_size;
            int count = min(tile_size, (int)(L - tile_start));
            int idx = tile_start + threadIdx.x;

            // Load one tile of data (apply mask) into shared memory
            scalar_t val = 0;
            if (threadIdx.x < count) {
                val = apply_mask_val(x_row[idx], mask_row[idx]);
            }
            tile_data[threadIdx.x] = val;
            __syncthreads();

            // Intra-tile inclusive scan (Hillis-Steele scan) on the valid elements
            for (int offset = 1; offset < count; offset *= 2) {
                scalar_t temp = 0;
                if (threadIdx.x >= offset && threadIdx.x < count) {
                    temp = tile_data[threadIdx.x - offset];
                }
                __syncthreads();
                if (threadIdx.x < count) {
                    tile_data[threadIdx.x] += temp;
                }
                __syncthreads();
            }

            // Write out the scanned tile with the running row offset added
            if (threadIdx.x < count) {
                out_row[idx] = tile_data[threadIdx.x] + row_offset;
            }
            
            // Update the running offset with the last element of this tile (only one thread does it)
            if (threadIdx.x == count - 1) {
                row_offset += tile_data[threadIdx.x];
            }
            __syncthreads();
        }
    }
}


// Macros to check inputs
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// Host function: sets up tensor shape, permutes, and launches the hybrid kernel
torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    CHECK_INPUT(x);
    CHECK_INPUT(mask);
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    // Adjust negative dims
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute dimensions to bring the target dimension to the last axis
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) {
            perm.push_back(i);
        }
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

    // Launch kernel: one block per row with 256 threads per block
    int threads = 256;
    int blocks = N;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hybrid_masked_cumsum_cuda", ([&] {
        hybrid_masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    // Reshape and permute output back to the original order
    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Hybrid Masked Cumulative Sum (CUDA)");
}
