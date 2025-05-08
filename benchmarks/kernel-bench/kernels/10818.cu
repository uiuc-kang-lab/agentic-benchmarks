#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// This hybrid kernel performs a masked cumulative sum along the last dimension of a tensor.
// It uses an efficient warp-level scan when the row length L is small (L <= blockDim.x),
// and for larger L it uses a block-level parallel scan across threads to distribute the work.

// Threshold is implicitly defined by blockDim.x. We assume blockDim.x is chosen to be 256.

// Device helper: get masked value
template <typename scalar_t>
__device__ __forceinline__ scalar_t get_val(const scalar_t* x, const bool* mask, int idx) {
    return mask[idx] ? x[idx] : static_cast<scalar_t>(0);
}

// Device function: warp-level masked inclusive scan for small L
// All threads in the block participate; threads with index >= L work on dummy zeros.
template <typename scalar_t>
__device__ void warp_masked_cumsum(const scalar_t* x_row, const bool* mask_row, scalar_t* out_row, int L) {
    int tid = threadIdx.x;
    int lane = tid & 31;          // lane index within a warp
    int warpId = tid >> 5;        // warp index within the block

    // Load value (if within bounds), else 0
    scalar_t val = (tid < L) ? get_val<scalar_t>(x_row, mask_row, tid) : static_cast<scalar_t>(0);

    // In-warp inclusive scan using shuffle intrinsics
    for (int offset = 1; offset < 32; offset <<= 1) {
        scalar_t y = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset)
            val += y;
    }

    // Allocate shared memory to hold the sum of each warp
    __shared__ scalar_t warpSums[32]; // Maximum 32 warps per block

    // Last thread in each warp writes its result to shared memory
    if ((lane == 31) || (tid == L - 1)) {
        warpSums[warpId] = val;
    }
    __syncthreads();

    // Each warp (except the first) adds the sum of all previous warps
    scalar_t warpOffset = 0;
    if (warpId > 0) {
        if (lane == 0) {
            for (int i = 0; i < warpId; i++) {
                warpOffset += warpSums[i];
            }
        }
        // Broadcast the offset to all threads in the warp
        warpOffset = __shfl_sync(0xffffffff, warpOffset, 0);
        val += warpOffset;
    }

    // Write the scanned value back if within bounds
    if (tid < L) {
        out_row[tid] = val;
    }
}

// Hybrid kernel: each block processes one row
// For L <= blockDim.x, uses warp-level scan; for L > blockDim.x, uses block-level parallel scan

template <typename scalar_t>
__global__ void hybrid_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    int row = blockIdx.x;  // one block per row
    if (row >= N) return;

    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* out_row = output + row * L;

    // If the row length is small enough, use warp-level scan
    if (L <= blockDim.x) {
        // All threads in the block participate in the warp scan
        warp_masked_cumsum<scalar_t>(x_row, mask_row, out_row, L);
    } else {
        // For larger rows, use block-level parallel scan.
        // Each thread processes a contiguous chunk of the row.
        int tid = threadIdx.x;
        int nThreads = blockDim.x;

        // Determine the chunk size for each thread
        int chunk = (L + nThreads - 1) / nThreads;
        int start = tid * chunk;
        int end = min(start + chunk, (int)L);

        // First pass: each thread computes the sum over its chunk
        scalar_t local_sum = 0;
        for (int j = start; j < end; j++) {
            local_sum += mask_row[j] ? x_row[j] : static_cast<scalar_t>(0);
        }

        // Allocate shared memory for partial sums
        extern __shared__ scalar_t sdata[];
        sdata[tid] = local_sum;
        __syncthreads();

        // Thread 0 computes the scan (offset) over the partial sums sequentially
        if (tid == 0) {
            scalar_t prefix = 0;
            for (int i = 0; i < nThreads; i++) {
                scalar_t temp = sdata[i];
                sdata[i] = prefix; // store offset for thread i
                prefix += temp;
            }
        }
        __syncthreads();

        // Each thread now has its offset stored in sdata[tid]
        scalar_t offset = sdata[tid];

        // Second pass: each thread computes a local prefix sum over its chunk and writes the results
        scalar_t prefix = offset;
        for (int j = start; j < end; j++) {
            scalar_t val = mask_row[j] ? x_row[j] : static_cast<scalar_t>(0);
            prefix += val;
            out_row[j] = prefix;
        }
    }
}

// Host function to set up and launch the hybrid CUDA kernel

torch::Tensor hybrid_masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    // Adjust negative dim
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

    // Reshape into 2D tensor: N rows and L columns
    int64_t L = x_permuted.size(-1);
    int64_t N = x_permuted.numel() / L;

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Launch one block per row. Use 256 threads per block.
    int threads = 256;
    dim3 block(threads);
    dim3 grid(N);

    // For the block-level scan branch, allocate shared memory of size (threads * sizeof(scalar_t)).
    size_t shared_mem = threads * sizeof(float);  // Note: This assumes x is float. For other types, sizeof(scalar_t) is used below.

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hybrid_masked_cumsum_cuda", ([&] {
        shared_mem = threads * sizeof(scalar_t);
        hybrid_masked_cumsum_kernel<scalar_t><<<grid, block, shared_mem>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    // Reshape and permute back to original shape
    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hybrid_masked_cumsum, "Hybrid Masked Cumulative Sum (CUDA)");
}
