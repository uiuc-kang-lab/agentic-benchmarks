#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel using shared memory and warp-level primitives
template <typename scalar_t>
__global__ void optimized_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    extern __shared__ scalar_t shared_mem[];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* out_row = output + row * L;

    // Process in chunks of WARP_SIZE to utilize warp-level operations
    constexpr int WARP_SIZE = 32;
    scalar_t sum = 0;
    
    // Use vectorized loads where possible
    using vec4_t = typename cuda::aligned_vector<scalar_t, 4>::type;
    
    #pragma unroll
    for (int64_t i = 0; i < L; i += WARP_SIZE) {
        int remaining = min(WARP_SIZE, static_cast<int>(L - i));
        
        // Load chunk into shared memory
        if (i + threadIdx.x % WARP_SIZE < L) {
            shared_mem[threadIdx.x % WARP_SIZE] = x_row[i + threadIdx.x % WARP_SIZE] * 
                static_cast<scalar_t>(mask_row[i + threadIdx.x % WARP_SIZE]);
        }
        
        // Warp-level parallel prefix sum
        #pragma unroll
        for (int offset = 1; offset < remaining; offset *= 2) {
            scalar_t val = __shfl_up_sync(0xffffffff, shared_mem[threadIdx.x % WARP_SIZE], offset);
            if (threadIdx.x % WARP_SIZE >= offset) {
                shared_mem[threadIdx.x % WARP_SIZE] += val;
            }
        }
        
        // Write results and update running sum
        if (i + threadIdx.x % WARP_SIZE < L) {
            out_row[i + threadIdx.x % WARP_SIZE] = sum + shared_mem[threadIdx.x % WARP_SIZE];
        }
        
        // Update running sum for next chunk
        sum += __shfl_sync(0xffffffff, shared_mem[min(remaining - 1, threadIdx.x % WARP_SIZE)], 
                          min(remaining - 1, WARP_SIZE - 1));
    }
}

torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    TORCH_CHECK(x.is_cuda() && mask.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && mask.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(x.sizes() == mask.sizes(), "Inputs must have same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "Mask must be boolean");

    dim = dim < 0 ? dim + x.dim() : dim;
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    auto perm = std::vector<int64_t>();
    for (int64_t i = 0; i < x.dim(); i++)
        if (i != dim) perm.push_back(i);
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const int shared_mem_size = 32 * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_masked_cumsum_cuda", ([&] {
        optimized_masked_cumsum_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); i++)
        inv_perm[perm[i]] = i;
        
    return output_permuted.permute(inv_perm);
}