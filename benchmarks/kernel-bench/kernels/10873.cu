#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void masked_cumsum_shared_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {
    
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shared_xm = reinterpret_cast<scalar_t*>(shared_mem);

    int row_idx = blockIdx.x;
    if (row_idx >= N) return;

    const scalar_t* x_row = x + row_idx * L;
    const bool* mask_row = mask + row_idx * L;
    scalar_t* output_row = output + row_idx * L;

    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        shared_xm[i] = mask_row[i] ? x_row[i] : scalar_t(0);
    }
    __syncthreads();

    for (int stride = 1; stride < L; stride *= 2) {
        for (int i = threadIdx.x; i < L; i += blockDim.x) {
            if (i >= stride) {
                shared_xm[i] += shared_xm[i - stride];
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        output_row[i] = shared_xm[i];
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
    const dim3 blocks(N);
    size_t shared_mem_size = L * sizeof(typename decltype(x_flat)::scalar_type);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_shared_cuda", ([&] {
        masked_cumsum_shared_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            N,
            L
        );
    }));

    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) inv_perm[perm[i]] = i;
    
    return output_permuted.permute(inv_perm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum (CUDA)");
}
