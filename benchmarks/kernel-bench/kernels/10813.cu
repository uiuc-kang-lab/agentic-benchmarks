#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Constant memory for frequently accessed configuration
__constant__ int64_t d_N;
__constant__ int64_t d_L;
__constant__ int d_warp_size = 32;
__constant__ int d_max_warps = 32;

template <typename scalar_t>
__global__ void const_mem_parallel_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output) {

    int row = blockIdx.x;
    if(row >= d_N) return;

    const scalar_t* x_row = x + row * d_L;
    const bool* mask_row = mask + row * d_L;
    scalar_t* out_row = output + row * d_L;

    // Use constant memory for frequently accessed values
    if(d_L <= blockDim.x) {
        int tid = threadIdx.x;
        int lane = tid & (d_warp_size - 1);
        int warpId = tid >> 5;

        // Load and mask data
        scalar_t val = static_cast<scalar_t>(0);
        if(tid < d_L) {
            val = mask_row[tid] ? x_row[tid] : static_cast<scalar_t>(0);
        }

        // Warp-level scan using constant memory for warp size
        #pragma unroll
        for (int offset = 1; offset < d_warp_size; offset <<= 1) {
            scalar_t y = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) {
                val += y;
            }
        }

        // Use constant memory for shared memory array size
        __shared__ scalar_t warpSums[32];  // Using d_max_warps

        // Store warp results
        if(lane == (d_warp_size - 1) || tid == d_L - 1) {
            warpSums[warpId] = val;
        }
        __syncthreads();

        // Accumulate across warps
        if(warpId > 0) {
            scalar_t warpOffset = 0;
            if(lane == 0) {
                #pragma unroll
                for (int i = 0; i < d_max_warps && i < warpId; i++) {
                    warpOffset += warpSums[i];
                }
            }
            warpOffset = __shfl_sync(0xffffffff, warpOffset, 0);
            val += warpOffset;
        }

        if(tid < d_L) {
            out_row[tid] = val;
        }
    } else {
        if(threadIdx.x == 0) {
            scalar_t sum = static_cast<scalar_t>(0);
            for (int64_t i = 0; i < d_L; i++) {
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

    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) {
            perm.push_back(i);
        }
    }
    perm.push_back(dim);
    
    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    int64_t L = x_permuted.size(-1);
    int64_t N = x_permuted.numel() / L;

    // Copy configuration to constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int64_t));
    cudaMemcpyToSymbol(d_L, &L, sizeof(int64_t));

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Configure kernel launch
    const int PARALLEL_THRESHOLD = 256;
    if (L <= PARALLEL_THRESHOLD) {
        int threads = 1;
        while (threads < L) threads *= 2;
        threads = std::min(threads, 1024);
        dim3 block(threads);
        dim3 grid(N);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
            const_mem_parallel_masked_cumsum_kernel<scalar_t><<<grid, block>>>(
                x_flat.data_ptr<scalar_t>(),
                mask_flat.data_ptr<bool>(),
                output_flat.data_ptr<scalar_t>()
            );
        }));
    } else {
        dim3 block(1);
        dim3 grid(N);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
            const_mem_parallel_masked_cumsum_kernel<scalar_t><<<grid, block>>>(
                x_flat.data_ptr<scalar_t>(),
                mask_flat.data_ptr<bool>(),
                output_flat.data_ptr<scalar_t>()
            );
        }));
    }

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