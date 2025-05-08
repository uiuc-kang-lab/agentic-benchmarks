/*
Hybrid Masked Cumulative Sum Kernel
This kernel combines efficient warp-level and block-level parallel scan approaches while also falling back to a serial scan for long rows.
It processes each row of the input 2D tensor (after permutation) independently. For rows with size L less or equal to a parallel threshold, it uses a parallel scan:
  - If the number of threads is <= warp size (32), it performs a warp-level scan using __shfl_up_sync.
  - Otherwise, it uses a block-level parallel scan via a shared-memory ping-pong algorithm.
For rows with L larger than the threshold, the kernel is launched with a single thread per row to avoid the overhead of parallelization.

Author: Expert CUDA Engineer
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Hybrid kernel: performs masked cumulative sum on each row

template <typename scalar_t>
__global__ void hybrid_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    // Each block processes one row
    int row = blockIdx.x;
    if (row >= N) return;

    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* out_row = output + row * L;

    int tid = threadIdx.x;
    const int WARP_SIZE = 32;

    // If more than one thread is available per row, use parallel scan
    if (blockDim.x > 1) {
        // When blockDim.x is small enough, use warp-level scan
        if (blockDim.x <= WARP_SIZE) {
            if (tid < L) {
                // Load element: apply mask
                scalar_t val = mask_row[tid] ? x_row[tid] : scalar_t(0);
                // Warp-level inclusive scan using shuffle
                for (int offset = 1; offset < blockDim.x; offset *= 2) {
                    scalar_t y = __shfl_up_sync(0xffffffff, val, offset);
                    if (tid >= offset) {
                        val += y;
                    }
                }
                out_row[tid] = val;
            }
        } else {
            // For larger blocks, use a block-level parallel scan with shared memory (ping-pong method)
            extern __shared__ char shared_mem[]; // dynamic shared memory as raw bytes
            scalar_t* s_scan = reinterpret_cast<scalar_t*>(shared_mem);
            scalar_t* s_temp = s_scan + blockDim.x; // second half of shared memory

            // Load input element with mask, if in bounds
            scalar_t myVal = (tid < L && mask_row[tid]) ? x_row[tid] : scalar_t(0);
            s_scan[tid] = myVal;
            __syncthreads();

            int n = blockDim.x;  // total threads in the block
            // Iterative scan: each step adds the value from an offset neighbor
            for (int offset = 1; offset < n; offset *= 2) {
                scalar_t add = (tid >= offset) ? s_scan[tid - offset] : scalar_t(0);
                __syncthreads();
                s_temp[tid] = s_scan[tid] + add;
                __syncthreads();
                s_scan[tid] = s_temp[tid];
            }
            
            if (tid < L) {
                out_row[tid] = s_scan[tid];
            }
        }
    } else {
        // Serial fallback: one-thread per row
        if (tid == 0) {
            scalar_t sum = scalar_t(0);
            for (int i = 0; i < L; ++i) {
                if (mask_row[i]) {
                    sum += x_row[i];
                }
                out_row[i] = sum;
            }
        }
    }
}


// Host function to prepare tensor dimensions, permutation and launch the kernel

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

    // Permute dimensions to bring the target dimension to the last position
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

    // Decide kernel launch configuration
    const int PARALLEL_THRESHOLD = 256; // tunable threshold
    int threads = 1;
    if (L <= PARALLEL_THRESHOLD) {
        // Choose number of threads: next power of two that is at least L (max 1024)
        threads = 1;
        while (threads < L) threads *= 2;
        threads = std::min(threads, 1024);
    } else {
        // For long rows, fallback to serial scan per row (threads = 1)
        threads = 1;
    }

    dim3 grid(N);
    dim3 block(threads);

    // Launch kernel using AT_DISPATCH to handle floating point types
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hybrid_masked_cumsum_cuda", ([&] {
        // Compute shared memory size: only needed when blockDim.x > warp size
        size_t shmem = (threads > 32 ? 2 * threads * sizeof(scalar_t) : 0);
        hybrid_masked_cumsum_kernel<scalar_t><<<grid, block, shmem>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N, L);
    }));

    // Reshape and invert permutation to restore original tensor layout
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
