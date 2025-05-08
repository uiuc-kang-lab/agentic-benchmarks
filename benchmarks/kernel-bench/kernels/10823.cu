#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel caches the entire row (or tiles of the row) in shared memory to reduce global memory accesses.
// It performs an inclusive masked cumulative sum using a Hillis-Steele scan algorithm in shared memory.
// For rows that fit entirely in one block (L <= blockDim.x), it loads the row into shared memory and scans it.
// For longer rows, it processes the row in tiles of size blockDim.x, using an extra shared memory variable to accumulate the offset between tiles.

// The kernel returns correct results without loss of precision by operating in the same precision as the input.

template <typename scalar_t>
__global__ void shared_cached_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    int row = blockIdx.x;
    if (row >= N) return;

    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* out_row = output + row * L;

    // Allocate shared memory for the scan array and one extra element for the running offset in tiled mode.
    // The dynamic shared memory size is (blockDim.x + 1) elements of type scalar_t.
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
    scalar_t* p_row_offset = sdata + blockDim.x;  // extra slot for the running offset

    int tid = threadIdx.x;

    // Case 1: The entire row fits in one block
    if (L <= blockDim.x) {
        if (tid < L) {
            sdata[tid] = mask_row[tid] ? x_row[tid] : static_cast<scalar_t>(0);
        } else {
            sdata[tid] = static_cast<scalar_t>(0);
        }
        __syncthreads();
        // Inclusive scan using Hillis-Steele algorithm
        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            scalar_t temp = (tid >= offset) ? sdata[tid - offset] : static_cast<scalar_t>(0);
            __syncthreads();
            sdata[tid] += temp;
            __syncthreads();
        }
        if (tid < L) {
            out_row[tid] = sdata[tid];
        }
    } else {
        // Case 2: Process the row in tiles of size blockDim.x
        // Initialize the running offset in shared memory (only one copy for the block)
        if (tid == 0) {
            p_row_offset[0] = static_cast<scalar_t>(0);
        }
        __syncthreads();
        
        int tile_size = blockDim.x;
        int num_tiles = (L + tile_size - 1) / tile_size; // ceiling division
        
        for (int t = 0; t < num_tiles; t++) {
            int idx = t * tile_size + tid;
            int count = ((L - t * tile_size) < tile_size) ? (L - t * tile_size) : tile_size;
            scalar_t val = static_cast<scalar_t>(0);
            if (tid < count) {
                val = mask_row[idx] ? x_row[idx] : static_cast<scalar_t>(0);
                sdata[tid] = val;
            } else {
                sdata[tid] = static_cast<scalar_t>(0);
            }
            __syncthreads();

            // Perform in-tile inclusive scan on the valid elements
            for (int offset = 1; offset < count; offset *= 2) {
                scalar_t temp = (tid >= offset && tid < count) ? sdata[tid - offset] : static_cast<scalar_t>(0);
                __syncthreads();
                if (tid < count)
                    sdata[tid] += temp;
                __syncthreads();
            }
            
            // Write out the scanned tile, adding the running offset
            if (tid < count) {
                out_row[idx] = sdata[tid] + p_row_offset[0];
            }
            __syncthreads();
            
            // Update the running offset with the last element of this tile
            if (tid == count - 1) {
                p_row_offset[0] += sdata[tid];
            }
            __syncthreads();
        }
    }
}


// Host function to prepare and launch the CUDA kernel

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

    // Adjust negative dimension
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

    // Reshape to 2D: N rows and L columns
    int64_t L = x_permuted.size(-1);
    int64_t N = x_permuted.numel() / L;

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Choose kernel launch configuration
    int threads = 0;
    // If the entire row fits in one block, use power-of-2 threads (max up to 1024)
    if (L <= 1024) {
        threads = 1;
        while (threads < L) {
            threads *= 2;
        }
        threads = std::min(threads, 1024);
    } else {
        // For longer rows use a fixed tile size (e.g., 256 threads per block)
        threads = 256;
    }

    dim3 grid(N);
    dim3 block(threads);
    // Allocate shared memory: (threads + 1) elements of type scalar_t
    size_t shared_mem = (threads + 1) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "shared_cached_masked_cumsum_cuda", ([&] {
        size_t shmem = (threads + 1) * sizeof(scalar_t);
        shared_cached_masked_cumsum_kernel<scalar_t><<<grid, block, shmem>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    // Reshape and permute the output back to the original layout
    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Shared Cached Masked Cumulative Sum (CUDA)");
}
