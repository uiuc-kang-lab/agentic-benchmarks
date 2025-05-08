#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Modular device function for performing masked cumulative sum for one row
template <typename scalar_t>
__device__ inline void compute_masked_cumsum(const scalar_t* __restrict__ x_row,
                                              const bool* __restrict__ mask_row,
                                              scalar_t* __restrict__ output_row,
                                              int64_t L) {
    scalar_t sum = scalar_t(0);
    for (int64_t i = 0; i < L; ++i) {
        if (mask_row[i]) {
            sum += x_row[i];
        }
        output_row[i] = sum;
    }
}

// Optimized kernel that uses a modular device function for each row
template <typename scalar_t>
__global__ void optimized_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Process one row using the modular device function
    compute_masked_cumsum<scalar_t>(x + idx * L,
                                      mask + idx * L,
                                      output + idx * L,
                                      L);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Host function that prepares tensors, launches the kernel, and reconstructs the output
torch::Tensor optimized_masked_cumsum(
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

    // Permute dimensions to move the target dimension to the last
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) {
            perm.push_back(i);
        }
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    // Reshape into a 2D tensor (N rows x L columns)
    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Launch the CUDA kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_masked_cumsum_kernel", ([&] {
        optimized_masked_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L);
    }));

    // Reshape and permute back to the original shape
    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_masked_cumsum, "Optimized Modular Masked Cumulative Sum (CUDA)");
}
