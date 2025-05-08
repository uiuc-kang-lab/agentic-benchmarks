#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constant memory for frequently accessed configuration
__constant__ int64_t d_N;
__constant__ int64_t d_L;
__constant__ int d_warp_size = 32;
__constant__ int d_max_threads = 256;
__constant__ int d_max_warps = 8;
__constant__ bool d_use_parallel;

// Helper function to compute next power of 2
__host__ __device__ inline int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

template <typename scalar_t>
__global__ void optimized_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output) {

    int row = blockIdx.x;
    if(row >= d_N) return;

    const scalar_t* x_row = x + row * d_L;
    const bool* mask_row = mask + row * d_L;
    scalar_t* out_row = output + row * d_L;

    if(d_use_parallel) {
        int tid = threadIdx.x;
        int lane = tid & (d_warp_size - 1);
        int warpId = tid >> 5;

        // Load data with mask applied
        scalar_t val = 0;
        if(tid < d_L) {
            val = mask_row[tid] ? x_row[tid] : 0;
        }

        // Warp-level scan using shuffle
        #pragma unroll
        for(int offset = 1; offset < d_warp_size; offset <<= 1) {
            scalar_t n = __shfl_up_sync(0xffffffff, val, offset);
            if(lane >= offset) {
                val += n;
            }
        }

        // Inter-warp communication through shared memory
        __shared__ scalar_t warp_results[8];  // Using constant d_max_warps

        if(lane == (d_warp_size - 1) || (tid == d_L - 1)) {
            warp_results[warpId] = val;
        }
        __syncthreads();

        // Add previous warps' sums
        if(warpId > 0 && tid < d_L) {
            scalar_t sum = 0;
            #pragma unroll
            for(int i = 0; i < d_max_warps && i < warpId; i++) {
                sum += warp_results[i];
            }
            val += sum;
        }

        // Write result
        if(tid < d_L) {
            out_row[tid] = val;
        }
    } else {
        // Sequential fallback for long rows
        if(threadIdx.x == 0) {
            scalar_t sum = 0;
            #pragma unroll 4
            for(int64_t i = 0; i < d_L; i++) {
                if(mask_row[i]) {
                    sum += x_row[i];
                }
                out_row[i] = sum;
            }
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

    int64_t L = x_permuted.size(-1);
    int64_t N = x_permuted.numel() / L;

    // Copy configuration to constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int64_t));
    cudaMemcpyToSymbol(d_L, &L, sizeof(int64_t));
    bool use_parallel = (L <= 256);
    cudaMemcpyToSymbol(d_use_parallel, &use_parallel, sizeof(bool));

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Configure kernel launch
    dim3 block(use_parallel ? std::min(256, next_power_of_2((int)L)) : 1);
    dim3 grid(N);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        optimized_masked_cumsum_kernel<scalar_t><<<grid, block>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>()
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
    m.def("forward", &masked_cumsum, "Optimized Masked Cumulative Sum with Constant Memory (CUDA)");
}