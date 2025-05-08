#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute the sum of squares for a given offset
__device__ float compute_sum_of_squares(const float* input, int start, int end, int stride_C, int base) {
    float thread_sum = 0.0f;
    for (int i = start; i < end; i += stride_C) {
        float val = input[base + i];
        thread_sum += val * val;
    }
    return thread_sum;
}

// Device function to perform the warp reduction
__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function to scale the vector
__device__ void scale_vector(float* output, const float* input, int start, int end, float inv_norm, int stride_C, int base) {
    for (int i = start; i < end; i += stride_C) {
        output[base + i] = input[base + i] * inv_norm;
    }
}

__global__ void l2norm_modular_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int base = vector_idx * outer_stride;
    const int tid = threadIdx.x;

    // Compute the sum of squares using a device function
    float thread_sum = compute_sum_of_squares(input, tid, C, stride_C, base);

    // Reduce within a warp
    thread_sum = warp_reduce_sum(thread_sum);

    // Use shared memory for block reduction
    __shared__ float shared_mem[32];
    if (tid % warpSize == 0) {
        shared_mem[tid / warpSize] = thread_sum;
    }
    __syncthreads();

    // Final reduction
    if (tid < 32) {
        float block_sum = (tid < 4) ? shared_mem[tid] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);

        // Normalize
        if (tid == 0) {
            shared_mem[0] = rsqrt(block_sum + 1e-12);
        }
    }
    __syncthreads();

    const float inv_norm = shared_mem[0];

    // Scale the vector using a device function
    scale_vector(output, input, tid, C, inv_norm, stride_C, base);
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    const dim3 blocks(total_vectors);
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_modular", ([&] {
        l2norm_modular_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
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