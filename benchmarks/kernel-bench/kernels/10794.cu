#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel using shared memory and warp-level primitives for reduction.
template <typename scalar_t>
__global__ void masked_cumsum_shared_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    extern __shared__ scalar_t shared_data[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warpId = tid / 32;

    if (row >= N) return;

    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* out_row = output + row * L;

    scalar_t sum = scalar_t(0);
    for (int i = tid; i < L; i += blockDim.x) {
        if (mask_row[i]) {
            sum += x_row[i];
        }
        shared_data[tid] = sum;
        __syncthreads();

        // Perform reduction within the warp
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Store the result for this warp's first thread
        if (lane == 0) {
            shared_data[warpId] = sum;
        }
        __syncthreads();

        // Final reduction across warps
        if (warpId == 0) {
            sum = (tid < blockDim.x / 32) ? shared_data[lane] : scalar_t(0);
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (tid == 0) {
                shared_data[0] = sum;
            }
        }
        __syncthreads();

        // Write the cumulative sum to output
        if (i < L) {
            out_row[i] = shared_data[0];
        }
    }
}

// Host function
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

    // Adjust dimension if negative
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute dimensions to bring the target dimension to the last
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim)
            perm.push_back(i);
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    // Reshape into 2D: each row corresponds to a reduction
    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    int blocks = N;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        masked_cumsum_shared_kernel<scalar_t><<<blocks, threadsPerBlock, threadsPerBlock * sizeof(scalar_t)>>> (
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
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
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum with Shared Memory and Warp Primitives (CUDA)");
}
