#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ void stream_aware_warp_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    cudaStream_t stream) {
    
    int out_idx = blockIdx.x;
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    scalar_t sum = 0;
    // Coalesced access pattern with stride equal to blockDim.x
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        sum += input[base + i * inner_size];
    }

    // Warp-level reduction (no shared memory)
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    if (threadIdx.x == 0) {
        // Asynchronous write with stream specification
        output[out_idx] = sum;
    }
}

torch::Tensor stream_pipelined_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Setup input tensor with async pinning
    auto input_options = input.options();
    if (!input.is_pinned()) {
        input = input.pinned_memory();
    }

    int64_t reduce_size = sizes[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= sizes[i];
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) inner_size *= sizes[i];

    sizes[dim] = 1;
    auto output = torch::empty(sizes, input_options.device(at::kCUDA));

    // Configure kernel launch parameters for H100
    const int64_t num_output = outer_size * inner_size;
    const int threads = 256; // Multiple warps for better occupancy
    const int blocks = (num_output + threads/WARP_SIZE - 1) / (threads/WARP_SIZE);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "stream_aware_reduce", ([&] {
        stream_aware_warp_reduce_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                reduce_size,
                inner_size,
                stream
            );
    }));

    // Synchronize only if necessary
    if (input_options.device().is_cpu()) {
        cudaStreamSynchronize(stream);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &stream_pipelined_reduce_cuda, "Stream-pipelined warp reduction (CUDA)");
}
