#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Constant memory for storing frequently accessed data
__constant__ bool const_mask[1024];  // Assuming max L <= 1024 for constant memory usage

// Kernel that uses constant memory for mask to improve memory access efficiency
// Each block processes one row, using warp-level scan when L <= threshold
#define PARALLEL_THRESHOLD 256

// Kernel: each block processes one row
// When using parallel scan, we assume blockDim.x is chosen as the next power-of-2 >= L (and <= PARALLEL_THRESHOLD).

template <typename scalar_t>
__global__ void constant_memory_cumsum_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    int row = blockIdx.x;  // one block per row
    if(row >= N) return;

    const scalar_t* x_row = x + row * L;
    scalar_t* out_row = output + row * L;

    // If L is small enough, use the warp-scan parallel algorithm
    if(L <= blockDim.x) {
        int tid = threadIdx.x;  // index within the block
        int lane = tid & 31;    // equivalent to tid % 32
        int warpId = tid >> 5;  // equivalent to tid / 32

        // Load element from global memory if within L, applying mask from constant memory
        scalar_t val = static_cast<scalar_t>(0);
        if(tid < L) {
            val = const_mask[tid] ? x_row[tid] : static_cast<scalar_t>(0);
        }

        // Perform an inclusive scan within the warp using shuffle intrinsics
        for (int offset = 1; offset < 32; offset <<= 1) {
            scalar_t y = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) {
                val += y;
            }
        }

        // Allocate shared memory to accumulate the sum from each warp
        __shared__ scalar_t warpSums[32];  // Maximum of 32 warps per block

        // Write the last element of each warp's scan into shared memory
        if(lane == 31 || tid == L - 1) {
            warpSums[warpId] = val;
        }
        __syncthreads();

        // Each warp, except the first, adds the sum of all previous warps to its results
        scalar_t warpOffset = 0;
        if(warpId > 0) {
            if(lane == 0) {
                for (int i = 0; i < warpId; i++) {
                    warpOffset += warpSums[i];
                }
            }
            warpOffset = __shfl_sync(0xffffffff, warpOffset, 0);
            val += warpOffset;
        }

        // Write the scanned value back to global memory if within bounds
        if(tid < L) {
            out_row[tid] = val;
        }

    } else {
        // Fallback: use a single thread (thread 0) to perform a sequential cumulative sum
        if(threadIdx.x == 0) {
            scalar_t sum = static_cast<scalar_t>(0);
            for (int64_t i = 0; i < L; i++) {
                if(const_mask[i]) {
                    sum += x_row[i];
                }
                out_row[i] = sum;
            }
        }
    }
}

// Host function to set up and launch the CUDA kernel

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

    // Adjust the dimension if negative
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

    // Reshape to 2D tensor: N rows and L columns
    int64_t L = x_permuted.size(-1);
    int64_t N = x_permuted.numel() / L;

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Copy mask to constant memory
    cudaMemcpyToSymbol(const_mask, mask_flat.data_ptr<bool>(), L * sizeof(bool));

    // Decide kernel configuration based on L
    if (L <= PARALLEL_THRESHOLD) {
        int threads = 1;
        while (threads < L) {
            threads *= 2;
        }
        threads = std::min(threads, 1024);
        dim3 block(threads);
        dim3 grid(N);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
            constant_memory_cumsum_kernel<scalar_t><<<grid, block>>>(
                x_flat.data_ptr<scalar_t>(),
                output_flat.data_ptr<scalar_t>(),
                N,
                L
            );
        }));
    } else {
        dim3 block(1);
        dim3 grid(N);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
            constant_memory_cumsum_kernel<scalar_t><<<grid, block>>>(
                x_flat.data_ptr<scalar_t>(),
                output_flat.data_ptr<scalar_t>(),
                N,
                L
            );
        }));
    }

    // Reshape and permute the output back to the original shape
    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum with Constant Memory (CUDA)");
}