#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_warps = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Process 4 elements per thread to increase ILP
    for (int group = 0; group < 4; group++) {
        const int linear_idx = gidx * 4 + group;
        const int batch_id = linear_idx / numel_per_batch;
        if (batch_id >= batch_size) return;

        const int offset_in_batch = linear_idx % numel_per_batch;
        const int batch_offset = batch_id * num_features * numel_per_batch;

        scalar_t sumsq = 0;
        #pragma unroll 8
        for (int feat = 0; feat < num_features; feat++) {
            const scalar_t val = input[batch_offset + feat * numel_per_batch + offset_in_batch];
            sumsq += val * val;
        }

        // Warp-reduce sum for better utilization
        for (int offset = 16; offset > 0; offset >>= 1)
            sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);

        if (lane_id == 0) {
            const scalar_t rms = rsqrt(sumsq / num_features + eps);
            #pragma unroll 8
            for (int feat = 0; feat < num_features; feat++) {
                const int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
                output[idx] = input[idx] * rms;
            }
        }
    }
}

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    int numel_per_batch = 1;
    for(int i = 2; i < input.dim(); i++)
        numel_per_batch *= input.size(i);

    const int total_threads = (batch_size * numel_per_batch + 3) / 4;
    const int threads_per_block = 128;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &rms_norm_cuda_forward, "Warp-optimized RMS normalization");
}
