#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel where each block handles a position (batch_id, element offset) and distributes the feature-dimension workload among threads.
// This ensures an even distribution of work across threads and blocks, reducing bottlenecks.

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    // Each block is responsible for one spatial position across the feature dimension.
    // Calculate the global position: pos in [0, batch_size * numel_per_batch)
    int pos = blockIdx.x;
    int batch_id = pos / numel_per_batch;
    int offset = pos % numel_per_batch;
    int base = batch_id * num_features * numel_per_batch;

    // Allocate shared memory for reduction (partial sums)
    extern __shared__ char shared[];  // raw shared memory
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared);

    // Each thread computes a partial sum over parts of the feature dimension
    scalar_t partial_sum = static_cast<scalar_t>(0);
    for (int feat = threadIdx.x; feat < num_features; feat += blockDim.x) { int idx = base + feat * numel_per_batch + offset; scalar_t val = input[idx];
        int idx = base + feat * numel_per_batch + offset;
        scalar_t val = input[idx];
        partial_sum += val * val;
    }

    // Store the partial sum in shared memory
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // After reduction, sdata[0] holds the sum of squares over the feature dimension
    scalar_t total = sdata[0];
    scalar_t rms = sqrt(total / static_cast<scalar_t>(num_features) + static_cast<scalar_t>(eps));

    // Broadcast rms to all threads (using shared memory)
    if (threadIdx.x == 0) sdata[0] = rms;
    __syncthreads();
    rms = sdata[0];

    // Now, each thread normalizes its portion of the feature dimension
    for (int feat = threadIdx.x; feat < num_features; feat += blockDim.x) { int idx = base + feat * numel_per_batch + offset; scalar_t val = input[idx];
        int idx = base + feat * numel_per_batch + offset;
        output[idx] = input[idx] / rms;
    }
}


torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Each block handles one (batch_id, offset) pair
    const int total_positions = batch_size * numel_per_batch;

    // Define the number of threads per block
    const int threads_per_block = 256;

    // Shared memory size in bytes
    size_t shared_mem_size = threads_per_block * sizeof(input.scalar_type());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<< total_positions, threads_per_block, threads_per_block * sizeof(scalar_t) >>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_features,
            numel_per_batch,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward, "RMS normalization forward (CUDA)");
}
