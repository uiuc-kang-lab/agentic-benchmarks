#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128

template <typename scalar_t>
__global__ void masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= N) return;
    
    const scalar_t* x_row = x + global_idx * L;
    const bool* mask_row = mask + global_idx * L;
    scalar_t* output_row = output + global_idx * L;

    scalar_t sum = 0;
    for (int64_t i = 0; i < L; ++i) {
        if (mask_row[i]) {
            sum += x_row[i];
        }
        output_row[i] = sum;
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

    if (dim < 0) dim += x.dim();
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    auto x_permuted = x.transpose(dim, -1).contiguous();
    auto mask_permuted = mask.transpose(dim, -1).contiguous();

    const int64_t N = x_permuted.numel() / x_permuted.size(-1);
    const int64_t L = x_permuted.size(-1);

    auto output_flat = torch::empty_like(x_permuted);

    const int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int threads = BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            x_permuted.data_ptr<scalar_t>(),
            mask_permuted.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    auto result = output_flat.transpose(dim, -1);
    return result.contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Optimized Masked Cumulative Sum (CUDA)");
}