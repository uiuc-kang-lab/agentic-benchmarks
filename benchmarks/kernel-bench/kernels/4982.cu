#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute partial sum of squares
template <typename scalar_t>
__device__ scalar_t compute_partial_sum(const scalar_t* __restrict__ input, int start, int end, int stride_C) {
    scalar_t sum = 0;
    for (int i = start; i < end; i += stride_C) {
        scalar_t val = input[i];
        sum += val * val;
    }
    return sum;
}

// Device function to perform reduction within a block
template <typename scalar_t>
__device__ scalar_t block_reduce_sum(scalar_t* shared_mem, scalar_t thread_sum) {
    int tid = threadIdx.x;
    shared_mem[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile scalar_t* smem = shared_mem;
        if (blockDim.x > 64) smem[tid] += smem[tid + 32];
        if (blockDim.x > 32) smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    return shared_mem[0];
}

// Device function to normalize the vector
template <typename scalar_t>
__device__ void normalize_vector(scalar_t* __restrict__ output, const scalar_t* __restrict__ input, scalar_t inv_norm, int start, int end, int stride_C) {
    for (int i = start; i < end; i += stride_C) {
        output[i] = input[i] * inv_norm;
    }
}

// Kernel using modular device functions
template <typename scalar_t>
__global__ void l2norm_modular_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int base = vector_idx * outer_stride;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Use shared memory for partial sums
    __shared__ scalar_t shared_mem[256];
    scalar_t thread_sum = compute_partial_sum(input + base, tid, C, stride);

    // Reduce within block
    scalar_t block_sum = block_reduce_sum(shared_mem, thread_sum);

    // Compute normalization factor
    if (tid == 0) {
        shared_mem[0] = rsqrt(block_sum + 1e-12);
    }
    __syncthreads();

    const scalar_t inv_norm = shared_mem[0];

    // Normalize using stride loops
    normalize_vector(output + base, input + base, inv_norm, tid, C, stride);
}

// Host function
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const dim3 blocks(total_vectors);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_modular", ([&] {
        l2norm_modular_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with modular device functions");
}
