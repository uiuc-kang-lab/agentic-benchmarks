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
    
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const int batch_offset = batch_idx * num_features * numel_per_batch;
    
    // Each warp handles consecutive elements in the batch
    scalar_t sumsq = 0.0f;
    
    // Stride by warp_size to ensure coalesced memory access
    for (int n = threadIdx.x; n < numel_per_batch; n += blockDim.x) {
        scalar_t thread_sumsq = 0.0f;
        
        // Each thread processes consecutive feature elements
        #pragma unroll 4
        for (int f = 0; f < num_features; f++) {
            const int idx = batch_offset + f * numel_per_batch + n;
            const scalar_t val = input[idx];
            thread_sumsq += val * val;
        }
        sumsq += thread_sumsq;
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sumsq += __shfl_down_sync(__activemask(), sumsq, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = sumsq;
    }
    __syncthreads();
    
    // First warp reduces results from all warps
    if (warp_id == 0 && lane_id < (blockDim.x / warp_size)) {
        sumsq = shared_mem[lane_id];
        
        #pragma unroll
        for (int offset = (blockDim.x/warp_size)/2; offset > 0; offset /= 2) {
            sumsq += __shfl_down_sync(__activemask(), sumsq, offset);
        }
        
        if (lane_id == 0) {
            shared_mem[0] = sumsq;
        }
    }
    __syncthreads();
    
    // Get final sum and calculate RMS
    const scalar_t final_sumsq = shared_mem[0];
    const scalar_t rms = sqrt(final_sumsq / (num_features * numel_per_batch) + eps);
    
    // Normalize with coalesced access
    for (int n = threadIdx.x; n < numel_per_batch; n += blockDim.x) {
        #pragma unroll 4
        for (int f = 0; f < num_features; f++) {
            const int idx = batch_offset + f * numel_per_batch + n;
            output[idx] = input[idx] / rms;
        }
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
    const int shared_mem_size = (threads_per_block / 32) * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<batch_size, threads_per_block, shared_mem_size>>>(
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