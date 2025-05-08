#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Threshold: use parallel scan for rows with length <= PARALLEL_THRESHOLD
#define PARALLEL_THRESHOLD 256

// Kernel: each block processes one row of L elements
// This kernel uses shared memory for intra-block operations and warp-level primitives
// (using __shfl_up_sync) for the final inter-warp prefix scan to compute warp offsets.
// Each thread loads one element, applies the mask, and then performs an intra-warp inclusive scan.
// The last thread of each warp writes its result to shared memory (warpSums).
// Then, the first warp computes an exclusive scan over warpSums using __shfl_up_sync,
// and each thread adds its warp's offset for the final result.

template <typename scalar_t>
__global__ void shared_warp_reduce_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    int row = blockIdx.x;  // one block per row
    if(row >= N) return;

    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* out_row = output + row * L;

    // Assume blockDim.x is chosen to cover L (when L <= PARALLEL_THRESHOLD).
    int tid = threadIdx.x;
    int lane = tid & 31;        // lane index in warp
    int warpId = tid >> 5;      // warp index within block

    // Load element and apply mask (if outside L, value remains 0).
    scalar_t val = static_cast<scalar_t>(0);
    if (tid < L) {
        val = mask_row[tid] ? x_row[tid] : static_cast<scalar_t>(0);
    }

    // Intra-warp inclusive scan using warp-level primitive __shfl_up_sync
    for (int offset = 1; offset < 32; offset *= 2) {
        scalar_t tmp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += tmp;
        }
    }

    // Shared memory to hold each warp's total (the last element in the warp's scan)
    __shared__ scalar_t warpSums[32];
    if ((lane == 31) || (tid == L - 1)) {
        warpSums[warpId] = val;
    }
    __syncthreads();

    // Inter-warp scan: the first warp computes exclusive prefix sums of warpSums
    if (tid < 32) {
        int numWarps = (L + 31) / 32;  // number of warps that contain valid data
        // Load original warp sum if in range, else 0
        scalar_t orig = (tid < numWarps) ? warpSums[tid] : static_cast<scalar_t>(0);
        scalar_t warp_val = orig;
        // Perform an inclusive scan among warpSums across threads in warp 0 using __shfl_up_sync
        for (int offset = 1; offset < numWarps; offset *= 2) {
            scalar_t y = __shfl_up_sync(0xffffffff, warp_val, offset);
            if (tid >= offset) {
                warp_val += y;
            }
        }
        // Convert to exclusive scan: for thread 0, prefix is 0; for others, subtract their own original value
        scalar_t prefix = (tid == 0) ? static_cast<scalar_t>(0) : (warp_val - orig);
        if (tid < numWarps) {
            warpSums[tid] = prefix;  // store the warp offset
        }
    }
    __syncthreads();

    // Add warp offset to each thread's intra-warp result
    if (warpId > 0 && tid < L) {
        val += warpSums[warpId];
    }

    // Write the final cumulative sum to global memory
    if (tid < L) {
        out_row[tid] = val;
    }
}


// Host function: sets up tensor dimensions and launches the kernel

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

    // Adjust target dimension if negative
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute dimensions to bring target dimension to the last
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

    // Launch kernel: for small L, use parallel scan; for larger L, fallback to sequential scan.
    if (L <= PARALLEL_THRESHOLD) {
        // Choose block size as the next power of 2 >= L (capped at 1024).
        int threads = 1;
        while (threads < L) {
            threads *= 2;
        }
        threads = std::min(threads, 1024);
        dim3 block(threads);
        dim3 grid(N);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
            shared_warp_reduce_cumsum_kernel<scalar_t><<<grid, block>>>(
                x_flat.data_ptr<scalar_t>(),
                mask_flat.data_ptr<bool>(),
                output_flat.data_ptr<scalar_t>(),
                N,
                L
            );
        }));
    } else {
        // Fallback: use one thread per row to perform sequential scan
        dim3 block(1);
        dim3 grid(N);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
            shared_warp_reduce_cumsum_kernel<scalar_t><<<grid, block>>>(
                x_flat.data_ptr<scalar_t>(),
                mask_flat.data_ptr<bool>(),
                output_flat.data_ptr<scalar_t>(),
                N,
                L
            );
        }));
    }

    // Reshape and permute output back to original shape
    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum with Shared Memory and Warp-level Reduction (CUDA)");
}
