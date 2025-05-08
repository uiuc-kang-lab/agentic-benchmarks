#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void l2_normalize_kernel_stage1(
    const scalar_t* input,
    scalar_t* partial_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    
    // Each thread accumulates its portion
    scalar_t thread_sum = 0.0;
    for (int k = threadIdx.x; k < C; k += blockDim.x) {
        const scalar_t val = input[base_offset + k * stride_C];
        thread_sum += val * val;
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // First thread in each warp has partial sum
    __shared__ scalar_t warp_sums[8];  // Assuming max 8 warps per block
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction across warps using first warp
    if (warp_id == 0 && lane_id < (blockDim.x >> 5)) {
        scalar_t warp_sum = warp_sums[lane_id];
        #pragma unroll
        for (int offset = (blockDim.x >> 6); offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (lane_id == 0) {
            partial_sums[vector_idx] = warp_sum;
        }
    }
}

template <typename scalar_t>
__global__ void l2_normalize_kernel_stage2(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* partial_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;
    const scalar_t inv_norm = 1.0 / (sqrt(partial_sums[vector_idx]) + 1e-12);

    // Vectorized memory access for better throughput
    #pragma unroll 4
    for (int k = threadIdx.x; k < C; k += blockDim.x) {
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
    auto partial_sums = torch::zeros({total_vectors}, input.options());

    const int threads = 256;  // Multiple of warp size (32)
    const int blocks = total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize", [&] {
        l2_normalize_kernel_stage1<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride
        );

        l2_normalize_kernel_stage2<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
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