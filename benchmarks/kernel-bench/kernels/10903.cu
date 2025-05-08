#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Device function for computing masked cumulative sum
__device__ float compute_masked_cumsum(const float* x_row, const bool* mask_row, float* output_row, int L) {
    float sum = 0;
    for (int i = 0; i < L; ++i) {
        if (mask_row[i]) {
            sum += x_row[i];
        }
        output_row[i] = sum;
    }
    return sum;
}

// CUDA kernel for masked cumulative sum
__global__ void masked_cumsum_kernel(
    const float* __restrict__ x,
    const bool* __restrict__ mask,
    float* __restrict__ output,
    int64_t N,
    int64_t L) {

    const int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    float* output_row = output + row * L;

    compute_masked_cumsum(x_row, mask_row, output_row, L);
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

    const int blocks = N;
    const int threads = BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        masked_cumsum_kernel<<<blocks, threads>>>(
            x_flat.data_ptr<float>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<float>(),
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
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum (CUDA)");
}