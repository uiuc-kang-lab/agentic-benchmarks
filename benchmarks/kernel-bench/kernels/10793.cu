#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel processes each row in parallel by dividing the row into contiguous chunks assigned to each thread.
// It performs a two-pass (partial-scan then adjustment) approach to compute the masked cumulative sum concurrently.

template <typename scalar_t>
__global__ void parallel_masked_cumsum_kernel(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t L) {

    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Divide row into contiguous chunks per thread
    int chunk_size = (L + num_threads - 1) / num_threads;
    int start = tid * chunk_size;
    int end = start + chunk_size;
    if (end > L) end = L;

    // Allocate shared memory for partial sums
    extern __shared__ scalar_t sdata[]; // size: num_threads * sizeof(scalar_t)

    // Step 1: Each thread computes the sum of its assigned chunk
    scalar_t local_sum = static_cast<scalar_t>(0);
    for (int j = start; j < end; j++) {
        bool m = mask[row * L + j];
        scalar_t val = m ? x[row * L + j] : static_cast<scalar_t>(0);
        local_sum += val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Step 2: Serial scan of the partial sums by thread 0
    if (tid == 0) {
        for (int i = 1; i < num_threads; i++) {
            sdata[i] += sdata[i - 1];
        }
    }
    __syncthreads();

    // Each thread's offset is the sum of all previous chunks
    scalar_t offset = (tid == 0) ? static_cast<scalar_t>(0) : sdata[tid - 1];

    // Step 3: Each thread performs a local cumulative sum for its chunk and writes the result
    local_sum = offset;
    for (int j = start; j < end; j++) {
        bool m = mask[row * L + j];
        scalar_t val = m ? x[row * L + j] : static_cast<scalar_t>(0);
        local_sum += val;
        output[row * L + j] = local_sum;
    }
}


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Host function wrapping the CUDA kernel launch
torch::Tensor parallel_masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    CHECK_INPUT(x);
    CHECK_INPUT(mask);
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    // Adjust negative dim
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute dimensions to bring the target dim to the last dimension
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim)
            perm.push_back(i);
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    // Reshape into 2D tensors: (N, L)
    int64_t L = x_permuted.size(-1);
    int64_t N = x_permuted.numel() / L;

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Launch configuration: one block per row, with fixed threads per block
    const int threads = 256;
    const int blocks = N;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "parallel_masked_cumsum_cuda", ([&] {
        parallel_masked_cumsum_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            x_flat.data_ptr<scalar_t>(),
            mask_flat.data_ptr<bool>(),
            output_flat.data_ptr<scalar_t>(),
            L
        );
    }));

    // Reshape and invert permutation to restore original dimensions
    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &parallel_masked_cumsum, "Parallel Masked Cumulative Sum (CUDA)");
}
