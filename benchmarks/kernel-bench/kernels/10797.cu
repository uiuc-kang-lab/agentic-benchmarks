#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void reduced_sync_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    
    extern __shared__ char shared_mem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(shared_mem);

    const int row = bid * (blockDim.x / 32) + warp_id;
    if (row >= N) return;

    if (L <= 32) {
        // Warp-level processing for short sequences - no synchronization needed
        const scalar_t* x_row = x + row * L;
        const bool* mask_row = mask + row * L;
        scalar_t* out_row = output + row * L;

        scalar_t val = 0;
        if (lane_id < L && mask_row[lane_id]) {
            val = x_row[lane_id];
        }

        #pragma unroll
        for (int offset = 1; offset <= 16; offset *= 2) {
            scalar_t n = __shfl_up_sync(0xffffffff, val, offset);
            if (lane_id >= offset) val += n;
        }

        if (lane_id < L) {
            out_row[lane_id] = val;
        }
    } else {
        // For longer sequences, use shared memory with minimal synchronization
        const int items_per_thread = (L + blockDim.x - 1) / blockDim.x;
        const scalar_t* x_row = x + row * L;
        const bool* mask_row = mask + row * L;
        scalar_t* out_row = output + row * L;
        
        // Load and compute partial sums with minimal synchronization
        scalar_t running_sum = 0;
        
        #pragma unroll
        for (int i = 0; i < items_per_thread; ++i) {
            const int idx = tid + i * blockDim.x;
            if (idx < L) {
                running_sum = mask_row[idx] ? running_sum + x_row[idx] : running_sum;
                shared[idx] = running_sum;
            }
        }
        
        // Single sync point to ensure all partial sums are visible
        __syncthreads();
        
        // Write results with coalesced access pattern
        #pragma unroll
        for (int i = 0; i < items_per_thread; ++i) {
            const int idx = tid + i * blockDim.x;
            if (idx < L) {
                out_row[idx] = shared[idx];
            }
        }
    }
}

torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_CUDA(mask);
    CHECK_CONTIGUOUS(mask);
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

    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (N + warps_per_block - 1) / warps_per_block;
    const int shared_mem_size = L * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        reduced_sync_masked_cumsum_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
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
    
    return output_permuted.permute(inv_perm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum with Reduced Sync (CUDA)");
}