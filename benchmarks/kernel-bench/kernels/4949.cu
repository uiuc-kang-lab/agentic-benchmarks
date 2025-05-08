#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void l2_normalize_kernel_stride(
    const scalar_t* input,
    scalar_t* output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;
    const int tid = threadIdx.x;

    scalar_t sum = 0.0;

    // Compute sum of squares with stride loop
    for (int k = tid; k < C; k += blockDim.x) {
        const scalar_t val = input[base_offset + k * stride_C];
        sum += val * val;
    }

    // Block-wise reduction
    __shared__ scalar_t shared_sum[256];
    const int lane = tid % 32;
    const int wid = tid / 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        shared_sum[wid] = sum;
    }
    __syncthreads();

    scalar_t block_sum = (tid < blockDim.x / 32) ? shared_sum[lane] : 0.0;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            shared_sum[0] = block_sum;
        }
    }
    __syncthreads();

    const scalar_t inv_norm = 1.0 / (sqrt(shared_sum[0]) + 1e-12);

    // Normalize with stride loop
    for (int k = tid; k < C; k += blockDim.x) {
        output[base_offset + k * stride_C] = input[base_offset + k * stride_C] * inv_norm;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize_stride", [&] {
        l2_normalize_kernel_stride<scalar_t><<<blocks, threads>>>(
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