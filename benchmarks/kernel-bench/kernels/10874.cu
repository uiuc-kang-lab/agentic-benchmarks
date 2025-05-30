#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for masked cumulative sum using shared memory
template <typename scalar_t>
__global__ void masked_cumsum_shared_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    extern __shared__ scalar_t shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx >= N * L)
        return;

    int row = idx / L;
    int col = idx % L;

    // Load data into shared memory
    shared_data[tid] = (col < L) ? x[idx] * mask[idx] : scalar_t(0);
    __syncthreads();

    // Perform cumulative sum in shared memory
    scalar_t sum = scalar_t(0);
    for (int i = 0; i <= col; ++i) {
        sum += shared_data[row * L + i];
    }

    // Write result to global memory
    output[idx] = sum;
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor masked_cumsum_shared(
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
    const int shared_mem_size = threads * sizeof(scalar_t); // Move this inside the AT_DISPATCH_FLOATING_TYPES block

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_shared_cuda", ([&] {
        masked_cumsum_shared_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &masked_cumsum_shared, "Masked Cumulative Sum with Shared Memory (CUDA)");
}