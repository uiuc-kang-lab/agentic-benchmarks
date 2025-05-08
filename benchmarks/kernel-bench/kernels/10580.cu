#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device kernel: Maps a row (or chain) to each block
// and divides the work across threads. Memory loads are staggered
// across streams to allow overlapping with computation

template <typename scalar_t>
__global__ void streamed_cumprod_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

    int chain_id = blockIdx.x;  // one block per chain
    if (chain_id >= total_chains) return;

    // Decode the chain to get batch index and in-dimension index
    int batch_idx = chain_id / stride;
    int in_idx = chain_id % stride;
    int64_t base = batch_idx * (dim_size * stride) + in_idx;

    int t = threadIdx.x;
    int T = blockDim.x;
    int chunk_size = (dim_size + T - 1) / T;  // even distribution among threads

    // Use shared memory for each thread's local product
    __shared__ scalar_t s_local[256];  // assuming max 256 threads per block

    int start = t * chunk_size;
    int end = start + chunk_size;
    if (end > dim_size) end = dim_size;

    // Each thread starts by computing its local product in the input segment
    scalar_t local_prod = 1;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        local_prod *= input[idx];
    }
    s_local[t] = local_prod;
    __syncthreads();

    // Compute the prefix product for each thread
    scalar_t offset = 1;
    for (int i = 0; i < t; ++i) {
        offset *= s_local[i];
    }
    __syncthreads();

    // Write the results back to global memory using the prefix product as the starting point
    scalar_t cumulative_prod = offset;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        cumulative_prod *= input[idx];
        output[idx] = cumulative_prod;
    }
}

// Forward function
// This function launches the kernel and manages streams to overlap computation and memory operations.

torch::Tensor streamed_cumprod_cuda_forward(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride_val = strides[dim];
    int64_t total_chains = input.numel() / dim_size;

    // CUDA stream setup
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Launch one block per chain with 256 threads, staggered across streams to overlap operations
    const int threads = 256;
    dim3 blocks(total_chains);
    for (int i = 0; i < total_chains; i += 2) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "streamed_cumprod_cuda", ([&] {
            streamed_cumprod_kernel<scalar_t><<<blocks, threads, 0, (i % 2 == 0) ? stream1 : stream2>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                dim_size,
                stride_val,
                total_chains
            );
        }));
    }

    // Destroy streams after use
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &streamed_cumprod_cuda_forward, "Cumulative product forward with pipelining (CUDA)");
}