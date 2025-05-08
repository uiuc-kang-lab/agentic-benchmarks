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
    extern __shared__ float shared_mem[];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = tid / numel_per_batch;
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_block = blockDim.x / 32;
    
    if (batch_id >= batch_size) return;
    
    const int offset_in_batch = tid % numel_per_batch;
    const int batch_offset = batch_id * num_features * numel_per_batch;

    // Calculate sum of squares using warp-level reduction
    scalar_t sumsq = 0.0f;
    
    #pragma unroll
    for (int feat = 0; feat < num_features; feat++) {
        const scalar_t val = input[batch_offset + feat * numel_per_batch + offset_in_batch];
        sumsq += val * val;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sumsq += __shfl_down_sync(__activemask(), sumsq, offset);
    }

    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = sumsq;
    }
    
    __syncthreads();

    // First warp reduces results from all warps
    if (warp_id == 0 && lane_id < warps_per_block) {
        sumsq = shared_mem[lane_id];
        
        #pragma unroll
        for (int offset = warps_per_block/2; offset > 0; offset /= 2) {
            sumsq += __shfl_down_sync(__activemask(), sumsq, offset);
        }
    }

    // Broadcast the final sum to all threads in the block
    if (warp_id == 0 && lane_id == 0) {
        shared_mem[0] = sumsq;
    }
    
    __syncthreads();
    
    // Calculate RMS
    const scalar_t rms = sqrt(shared_mem[0] / num_features + eps);
    
    // Normalize
    #pragma unroll
    for (int feat = 0; feat < num_features; feat++) {
        const int idx = batch_offset + feat * numel_per_batch + offset_in_batch;
        output[idx] = input[idx] / rms;
    }
}

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    int numel_per_batch = 1;
    for(int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    const int threads_per_block = 256;
    const int blocks = (batch_size * numel_per_batch + threads_per_block - 1) / threads_per_block;
    const int shared_mem_size = (threads_per_block / 32) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
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