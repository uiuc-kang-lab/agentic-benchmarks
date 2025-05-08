#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// CUDA kernel for masked cumulative sum over a subset of rows using streams
template <typename scalar_t>
__global__ void masked_cumsum_kernel_stream(
    const scalar_t* __restrict__ x,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t start,
    int64_t num_rows,
    int64_t L) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows)
        return;

    int64_t row = start + idx;
    const scalar_t* x_row = x + row * L;
    const bool* mask_row = mask + row * L;
    scalar_t* output_row = output + row * L;

    scalar_t sum = scalar_t(0);
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

// The forward function that partitions the input for pipelined execution
torch::Tensor masked_cumsum(
    const torch::Tensor& x,
    const torch::Tensor& mask,
    int64_t dim) {

    CHECK_INPUT(x);
    CHECK_INPUT(mask);
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be a boolean tensor");

    // Adjust dim to be non-negative
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension");

    // Permute dimensions to bring the target dim to the last
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim)
            perm.push_back(i);
    }
    perm.push_back(dim);

    auto x_permuted = x.permute(perm).contiguous();
    auto mask_permuted = mask.permute(perm).contiguous();

    // Reshape to 2D tensors
    int64_t N = x_permuted.numel() / x_permuted.size(-1);
    int64_t L = x_permuted.size(-1);

    auto x_flat = x_permuted.view({N, L});
    auto mask_flat = mask_permuted.view({N, L});
    auto output_flat = torch::empty_like(x_flat);

    // Parameters for launching kernel asynchronously
    const int threads = 256;
    // Choose number of streams (use 4 streams or fewer if N is small)
    int num_streams = 4;
    if (N < num_streams) {
        num_streams = static_cast<int>(N);
    }
    int64_t chunk_size = (N + num_streams - 1) / num_streams;

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch the kernel on each stream for its chunk of rows
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "masked_cumsum_cuda", ([&] {
        for (int i = 0; i < num_streams; i++) {
            int64_t start_row = i * chunk_size;
            int64_t rows_in_chunk = std::min(chunk_size, N - start_row);
            if (rows_in_chunk <= 0) break;
            int blocks = (rows_in_chunk + threads - 1) / threads;
            masked_cumsum_kernel_stream<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                x_flat.data_ptr<scalar_t>(),
                mask_flat.data_ptr<bool>(),
                output_flat.data_ptr<scalar_t>(),
                start_row,
                rows_in_chunk,
                L
            );
        }
    }));

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Reshape and permute back to original shape
    auto output_permuted = output_flat.view(x_permuted.sizes());
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    auto output = output_permuted.permute(inv_perm);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &masked_cumsum, "Masked Cumulative Sum with Stream Pipelining (CUDA)");
}
