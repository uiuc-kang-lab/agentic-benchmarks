#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Hybrid implementation combining branchless operations with shared memory
template <typename scalar_t>
__global__ void hybrid_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    extern __shared__ char shared_mem[];
    scalar_t* shared_sums = reinterpret_cast<scalar_t*>(shared_mem);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const scalar_t* x_row = x + idx * L;
    const bool* mask_row = mask + idx * L;
    scalar_t* output_row = output + idx * L;

    scalar_t sum = 0;
    
    // Process in tiles to utilize shared memory
    const int TILE_SIZE = 32;
    for (int tile = 0; tile < (L + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int start = tile * TILE_SIZE;
        int end = min(start + TILE_SIZE, L);
        
        // Load tile data into shared memory and compute partial sums
        if (threadIdx.x < (end - start)) {
            shared_sums[threadIdx.x] = x_row[start + threadIdx.x] * 
                                     static_cast<scalar_t>(mask_row[start + threadIdx.x]);
        }
        __syncthreads();
        
        // Compute cumulative sum for this tile
        for (int i = start; i < end; ++i) {
            sum += shared_sums[i - start];
            output_row[i] = sum;
        }
        __syncthreads();
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor hybrid_masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    CHECK_INPUT(x);
    CHECK_INPUT(mask);
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

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const int shared_mem_size = 32 * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hybrid_masked_cumsum_cuda", ([&] {
        hybrid_masked_cumsum_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &hybrid_masked_cumsum, "Hybrid Masked Cumulative Sum (CUDA)");
}