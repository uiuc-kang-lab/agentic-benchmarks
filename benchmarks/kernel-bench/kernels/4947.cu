#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int d_C;
__constant__ int d_total_vectors;
__constant__ int d_stride_C;
__constant__ int d_outer_stride;

template <typename scalar_t>
__global__ void l2_normalize_kernel(
    const scalar_t* input,
    scalar_t* output) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= d_total_vectors) return;

    const int base_offset = vector_idx * d_outer_stride;

    scalar_t sum = 0.0;

    // Compute sum of squares
    for (int k = threadIdx.x; k < d_C; k += blockDim.x) {
        const scalar_t val = input[base_offset + k * d_stride_C];
        sum += val * val;
    }

    // Block-wise reduction
    __shared__ scalar_t shared_sum[256];
    scalar_t warp_sum = 0.0;
    const int lane = threadIdx.x % 32;
    const int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        shared_sum[wid] = sum;
    }
    __syncthreads();

    scalar_t block_sum = (threadIdx.x < blockDim.x / 32) ? shared_sum[lane] : 0.0;
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

    // Normalize and write output
    for (int k = threadIdx.x; k < d_C; k += blockDim.x) {
        output[base_offset + k * d_stride_C] = input[base_offset + k * d_stride_C] * inv_norm;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    // Copy constants to device constant memory
    cudaMemcpyToSymbol(d_C, &C, sizeof(int));
    cudaMemcpyToSymbol(d_total_vectors, &total_vectors, sizeof(int));
    cudaMemcpyToSymbol(d_stride_C, &stride_C, sizeof(int));
    cudaMemcpyToSymbol(d_outer_stride, &outer_stride, sizeof(int));

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize", [&] {
        l2_normalize_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>()
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization along dim=1");
}