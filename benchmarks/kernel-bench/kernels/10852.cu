#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

template <typename scalar_t>
__global__ void warp_optimized_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    extern __shared__ char shared_mem[];
    scalar_t* s_partial_sums = (scalar_t*)shared_mem;
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int wid = tid / WARP_SIZE;
    
    if (row >= N) return;

    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* output_row = output + row * L;

    // Process elements in warps
    scalar_t warp_sum = 0;
    
    scalar_t thread_sum = 0;
    
    // Each thread processes its assigned elements
    for (int i = tid; i < L; i += BLOCK_SIZE) {
        scalar_t val = (mask_row[i]) ? x_row[i] : 0;
        thread_sum = val;
        
        // Perform exclusive scan within warp
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            scalar_t n = __shfl_up_sync(0xffffffff, thread_sum, offset);
            if (lane >= offset) {
                thread_sum += n;
            }
        }
        
        // Last thread in each warp saves partial sum
        if (lane == WARP_SIZE - 1) {
            s_partial_sums[wid] = thread_sum;
        }
        __syncthreads();
        
        // First warp processes partial sums
        if (wid == 0 && lane < WARPS_PER_BLOCK) {
            scalar_t partial = s_partial_sums[lane];
            
            // Exclusive scan of partial sums
            #pragma unroll
            for (int offset = 1; offset < WARPS_PER_BLOCK; offset *= 2) {
                scalar_t n = __shfl_up_sync(0xffffffff, partial, offset);
                if (lane >= offset) {
                    partial += n;
                }
            }
            s_partial_sums[lane] = partial;
        }
        __syncthreads();
        
        // Add prefix from previous warps
        if (wid > 0) {
            thread_sum += s_partial_sums[wid - 1];
        }
        
        // Write result
        if (i < L) {
            output_row[i] = thread_sum;
        }
    }
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor warp_optimized_masked_cumsum(
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

    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim)
            perm.push_back(i);
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    const int threads = BLOCK_SIZE;
    const int blocks = N;
    const size_t shared_mem_size = WARPS_PER_BLOCK * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "warp_optimized_masked_cumsum_cuda", ([&] {
        warp_optimized_masked_cumsum_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    auto output = output_permuted.permute(inv_perm);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_optimized_masked_cumsum, "Warp Optimized Masked Cumulative Sum (CUDA)");
}