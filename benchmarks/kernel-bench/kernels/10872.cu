#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void masked_cumsum_block_scan_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    extern __shared__ scalar_t shared_sums[];
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row_idx >= N) return;

    const scalar_t* x_row = x + row_idx * L;
    const bool* mask_row = mask + row_idx * L;
    scalar_t* output_row = output + row_idx * L;

    scalar_t local_sum = 0;
    const int elements_per_thread = (L + blockDim.x - 1) / blockDim.x;
    const int start = tid * elements_per_thread;
    const int end = min(start + elements_per_thread, L);

    scalar_t carry = 0;
    for (int i = start; i < end; ++i) {
        if (mask_row[i]) carry += x_row[i];
        output_row[i] = carry;
        local_sum = carry;
    }

    shared_sums[tid] = local_sum;
    __syncthreads();

    // Block-wide exclusive scan
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        scalar_t temp = 0;
        if (tid >= stride)
            temp = shared_sums[tid - stride];
        __syncthreads();
        if (tid >= stride)
            shared_sums[tid] += temp;
        __syncthreads();
    }

    const scalar_t prev_sum = (tid == 0) ? 0 : shared_sums[tid - 1];
    
    for (int i = start; i < end; ++i) {
        if (i < L)
            output_row[i] += prev_sum - (mask_row[i] ? x_row[i] : 0);
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

    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i)
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
    const size_t shared_mem = threads * x_flat.element_size();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_block_scan_cuda", ([&] {
        masked_cumsum_block_scan_kernel<scalar_t><<<N, threads, shared_mem>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i)
        inv_perm[perm[i]] = i;
    return output_permuted.permute(inv_perm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum (CUDA)");
}