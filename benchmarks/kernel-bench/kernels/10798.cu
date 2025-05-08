#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel leveraging shared memory to perform masked cumulative sum

template <typename scalar_t>
__global__ void shared_mem_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t N,
    int64_t L) {

    extern __shared__ scalar_t shared_data[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // Load each row into shared memory
    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* out_row = output + row * L;

    scalar_t sum = scalar_t(0);
    for (int64_t offset = 0; offset < L; offset += blockDim.x) {
        int idx = offset + threadIdx.x;
        if (idx < L) {
            shared_data[threadIdx.x] = mask_row[idx] ? x_row[idx] : scalar_t(0);
        } else {
            shared_data[threadIdx.x] = scalar_t(0);
        }
        __syncthreads();

        // Calculate cumulative sum in shared memory
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            scalar_t val = 0;
            if (threadIdx.x >= stride) {
               val = shared_data[threadIdx.x - stride];
            }
            __syncthreads();  // Ensure all threads have updated the shared memory

            shared_data[threadIdx.x] += val;
            __syncthreads();  // Wait for partial sums to complete
        }

        // Write results from shared memory back to global memory
        if (idx < L) {
            sum += shared_data[threadIdx.x];
            out_row[idx] = sum;
        }
        __syncthreads();
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

    // Determine kernel launch configuration
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t shared_memory_size = threadsPerBlock * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        shared_mem_cumsum_kernel<scalar_t><<<blocks, threadsPerBlock, shared_memory_size>>>(
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
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum with Shared Memory (CUDA)");
}
