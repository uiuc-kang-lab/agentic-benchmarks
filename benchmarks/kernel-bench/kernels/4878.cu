#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void l1_norm_forward_kernel_coalesced(const float* __restrict__ x,
                                                float* __restrict__ out,
                                                int N,
                                                int D) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warps_per_block = blockDim.x >> 5;
    
    // Initialize partial sum for this thread
    float sum = 0.0f;
    
    // Calculate starting position for this thread
    int base_idx = row * D + lane_id;
    const int stride = 32; // Warp size for coalesced access
    
    // Process elements with coalesced access pattern
    for (int col = lane_id; col < D; col += stride) {
        float val = x[base_idx];
        sum += fabsf(val);
        base_idx += stride;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < warps_per_block) {
        float warp_sum = sdata[tid];
        #pragma unroll
        for (int offset = warps_per_block >> 1; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (tid == 0) {
            sdata[0] = (warp_sum > 0.0f) ? warp_sum : 1e-12f;
        }
    }
    __syncthreads();
    
    const float total_sum = sdata[0];
    
    // Normalize with coalesced access pattern
    base_idx = row * D + lane_id;
    #pragma unroll 4
    for (int col = lane_id; col < D; col += stride) {
        if (col < D) {
            float val = x[base_idx];
            out[base_idx] = val / total_sum;
        }
        base_idx += stride;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
    x = x.contiguous();
    
    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);
    
    // Calculate optimal thread block size
    const int threads_per_block = 256; // Multiple of warp size
    const int warps_per_block = threads_per_block >> 5;
    const int shared_mem_size = warps_per_block * sizeof(float);
    
    l1_norm_forward_kernel_coalesced<<<N, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA with coalesced memory access)");
}