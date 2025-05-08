#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

constexpr int WARP_SIZE = 32;

// Kernel using warp-level primitives (__shfl_up_sync) for cumulative sum
// Each block processes one row using a single warp (32 threads), and the row is processed in segments of WARP_SIZE elements

template <typename scalar_t>
__global__ void masked_cumsum_warp_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    int row = blockIdx.x;  // one block per row
    if (row >= N) return;

    int lane = threadIdx.x;  // lane: 0 to WARP_SIZE-1 (expects blockDim.x == WARP_SIZE)
    int64_t row_offset_idx = row * L;
    
    // prefix holds the total sum up to the current segment
    __shared__ scalar_t prefix[WARP_SIZE]; prefix[threadIdx.x] = 0; __syncthreads();

    // Number of segments needed to cover the row
    int num_segments = (L + WARP_SIZE - 1) / WARP_SIZE;

    for (int seg = 0; seg < num_segments; seg++) {
        int idx = seg * WARP_SIZE + lane;  // global index in the row
        scalar_t val = 0;
        if (idx < L) {
            // Load the element conditionally if mask is true
            bool m = mask[row_offset_idx + idx];
            val = m ? x[row_offset_idx + idx] : 0;
        }

        // Perform warp-level inclusive scan on the current segment
        // Each thread computes the sum of all values preceding it in this warp
        scalar_t sum_val = val;
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            scalar_t n = __shfl_up_sync(0xFFFFFFFF, sum_val, offset);
            if (lane >= offset) sum_val += n;
        }

        // Add the offset from previous segments
        sum_val += prefix;

        // Write the result for this element if valid
        if (idx < L) {
            output[row_offset_idx + idx] = sum_val;
        }

        // Determine the total sum of the current segment.
        // For a full segment, the last thread (lane = WARP_SIZE - 1) holds the total.
        // For a partial segment (last segment), compute the last active lane
        int last_lane = (seg == num_segments - 1) ? ((L - 1) % WARP_SIZE) : (WARP_SIZE - 1);
        // Broadcast the segment total from the designated lane
        scalar_t seg_total = __shfl_sync(0xFFFFFFFF, sum_val, last_lane);
        // Update prefix to be used in the next segment
        prefix = seg_total;
    }
}


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    CHECK_INPUT(x);
    CHECK_INPUT(mask);
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute so that target dimension is the last dimension
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); i++) {
        if (i != dim) {
            perm.push_back(i);
        }
    }
    perm.push_back(dim);

    auto x_perm = x.permute(perm).contiguous();
    auto mask_perm = mask.permute(perm).contiguous();

    int64_t N = x_perm.numel() / x_perm.size(-1);
    int64_t L = x_perm.size(-1);

    auto x_flat = x_perm.view({N, L});
    auto mask_flat = mask_perm.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Launch one warp per row: gridDim = N, blockDim = WARP_SIZE
    const int threads = WARP_SIZE;
    const int blocks = N;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        masked_cumsum_warp_kernel<scalar_t><<<blocks, threads>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    auto output_perm = output_flat.view(x_perm.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_perm.permute(inv_perm);
    return output.contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Warp-level Masked Cumulative Sum (CUDA)");
}
