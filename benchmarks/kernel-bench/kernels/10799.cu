#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel that processes one row per block using shared memory to ensure coalesced global memory accesses.
// Each thread in the block cooperatively loads a contiguous chunk of the row into shared memory,
// then a single thread computes the cumulative sum sequentially in shared memory, and finally threads write
// the result back to global memory in a coalesced fashion.

template <typename scalar_t>
__global__ void coalesced_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t L) {
    // Allocate shared memory for one entire row
    extern __shared__ scalar_t sdata[];

    // Each block processes one row
    int row = blockIdx.x;
    int row_offset = row * L;

    // Coalesced load from global memory into shared memory
    for (int j = threadIdx.x; j < L; j += blockDim.x) {
        int idx = row_offset + j;
        sdata[j] = mask[idx] ? x[idx] : scalar_t(0);
    }
    __syncthreads();

    // Compute sequential cumulative sum in shared memory (preserving the exact order of addition)
    if (threadIdx.x == 0) {
        for (int j = 1; j < L; ++j) {
            sdata[j] += sdata[j - 1];
        }
    }
    __syncthreads();

    // Coalesced store from shared memory back to global memory
    for (int j = threadIdx.x; j < L; j += blockDim.x) {
        int idx = row_offset + j;
        output[idx] = sdata[j];
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Host function to launch the CUDA kernel
torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    CHECK_INPUT(x);
    CHECK_INPUT(mask);
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    // Adjust dimension if negative
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute dimensions to bring the target dimension to the last position
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) {
            perm.push_back(i);
        }
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    // Reshape tensors to 2D: N rows and L columns
    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Launch one block per row. Use min(L, 256) threads per block to load/store in a coalesced way.
    int threads = (L < 256) ? L : 256;
    int blocks = N;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        size_t shared_mem_bytes = L * sizeof(scalar_t);
        coalesced_masked_cumsum_kernel<scalar_t><<<blocks, threads, shared_mem_bytes>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            L
        );
    }));

    // Reshape and permute back to the original tensor shape
    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Coalesced Masked Cumulative Sum (CUDA)");
}
