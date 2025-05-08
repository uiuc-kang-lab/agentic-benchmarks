#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void uniform_flow_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* output_row = output + row * L;

    scalar_t sum = 0;
    
    // Process in chunks of 32 elements (warp size) for better memory coalescing
    constexpr int CHUNK_SIZE = 32;
    scalar_t chunk_values[CHUNK_SIZE];
    
    #pragma unroll
    for (int64_t base = 0; base < L; base += CHUNK_SIZE) {
        // Pre-load chunk values and convert mask to multiplier
        #pragma unroll
        for (int i = 0; i < CHUNK_SIZE; i++) {
            int64_t idx = base + i;
            if (idx < L) {
                // Convert boolean mask to multiplier (0 or 1) to avoid branching
                scalar_t multiplier = static_cast<scalar_t>(mask_row[idx]);
                chunk_values[i] = x_row[idx] * multiplier;
            } else {
                chunk_values[i] = 0;
            }
        }

        // Compute running sum for the chunk
        #pragma unroll
        for (int i = 0; i < CHUNK_SIZE; i++) {
            int64_t idx = base + i;
            if (idx < L) {
                sum += chunk_values[i];
                output_row[idx] = sum;
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

    // Launch kernel with optimal thread configuration
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        uniform_flow_cumsum_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum (CUDA)");
}