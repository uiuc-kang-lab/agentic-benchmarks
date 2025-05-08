#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void l2_normalize_kernel_optimized(
    const scalar_t* input,
    scalar_t* output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;

    scalar_t sum = 0.0;

    // Compute sum of squares
    for (int k = threadIdx.x; k < C; k += blockDim.x) {
        const scalar_t val = input[base_offset + k * stride_C];
        sum += val * val;
    }

    // Block-wise reduction using shared memory
    __shared__ scalar_t shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within block using balanced binary tree
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Normalize and write output
    if (threadIdx.x == 0) {
        const scalar_t inv_norm = 1.0 / (sqrt(shared_sum[0]) + 1e-12);
        for (int k = 0; k < C; ++k) {
            output[base_offset + k * stride_C] = input[base_offset + k * stride_C] * inv_norm;
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0); // Simplified assumption for contiguous tensors

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (total_vectors + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize", [&] {
        l2_normalize_kernel_optimized<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization along dim=1");
}