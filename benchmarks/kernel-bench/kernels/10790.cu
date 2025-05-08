#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for masked cumulative sum along the last dimension
template <typename scalar_t>
__global__ void masked_cumsum_kernel_optimized(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    extern __shared__ scalar_t shared_sum[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx >= N * L)
        return;

    int row = idx / L;
    int col = idx % L;

    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* output_row = output + row * L;

    // Initialize shared memory
    shared_sum[tid] = 0;

    // Load data into shared memory
    scalar_t val = 0;
    if (mask_row[col]) {
        val = x_row[col];
    }

    shared_sum[tid] = val;

    // Synchronize to ensure all threads have written to shared memory
    __syncthreads();

    // Perform sequential scan in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        scalar_t temp = 0;
        if (tid >= stride) {
            temp = shared_sum[tid - stride];
        }
        __syncthreads();
        shared_sum[tid] += temp;
        __syncthreads();
    }

    // Write result to global memory
    output_row[col] = shared_sum[tid];
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

    // Adjust dim to be non-negative
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute dimensions to bring the target dim to the last
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim)
            perm.push_back(i);
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    // Reshape to 2D tensors
    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Launch CUDA kernel
    const int threads = 256;
    const int blocks = (N * L + threads - 1) / threads;
    size_t shared_memory_size = threads * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        masked_cumsum_kernel_optimized<scalar_t><<<blocks, threads, shared_memory_size>>>(
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
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum (CUDA)");
}