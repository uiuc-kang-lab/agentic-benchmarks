#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// This kernel assigns each thread a contiguous chunk of the vector so that the workload is evenly distributed.
// Each block processes one vector (e.g. along dim=0), and threads within the block compute partial sums of squares over their assigned chunk.
// A shared memory reduction then yields the L2 norm, which is used by each thread to normalize its segment.

template <typename scalar_t>
__global__ void l2_norm_even_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,              // number of elements per vector
    const int outer_stride,   // stride between vectors
    const int stride_C) {     // stride for each element (may be > 1 for non-contiguous tensors)

    // Each block is responsible for one vector
    int vector_idx = blockIdx.x;
    int base = vector_idx * outer_stride;

    // Determine the workload chunk per thread
    int chunk = (C + blockDim.x - 1) / blockDim.x;  // ceiling division
    int start = threadIdx.x * chunk;
    int end = start + chunk;
    if (end > C) end = C;

    // Each thread computes the sum of squares for its assigned chunk
    scalar_t local_sum = 0;
    for (int i = start; i < end; i++) {
        int idx = base + i * stride_C;
        scalar_t val = input[idx];
        local_sum += val * val;
    }

    // Use shared memory for block-level reduction
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread computes the normalization factor for the entire vector
    scalar_t total_sum = sdata[0];
    scalar_t norm = sqrt(total_sum) + static_cast<scalar_t>(1e-12);
    scalar_t inv_norm = static_cast<scalar_t>(1.0) / norm;

    // Each thread normalizes its chunk
    for (int i = start; i < end; i++) {
        int idx = base + i * stride_C;
        output[idx] = input[idx] * inv_norm;
    }
}

// Forward function

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    // Assume input shape is [N, C] or equivalent
    const int total_vectors = input.size(0);
    const int C = input.size(1);
    const int outer_stride = input.stride(0);
    const int stride_C = input.stride(1);

    auto output = torch::empty_like(input);

    const int threads = 256;
    // One block per vector
    // Allocate shared memory dynamically: one scalar_t per thread
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_even", ([&] {
        l2_norm_even_kernel<scalar_t><<<total_vectors, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C,
            outer_stride,
            stride_C
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with even workload distribution across threads");
}
