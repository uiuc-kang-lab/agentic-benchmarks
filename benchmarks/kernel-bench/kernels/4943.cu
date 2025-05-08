#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void l1_norm_opt_coalesced_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N,
    int D) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_size = 32;
    const int lane_id = tid % warp_size;
    
    __shared__ float smem[32]; // Shared memory for warp-level reduction
    
    const float4* x_vec = reinterpret_cast<const float4*>(x + row * D);
    const int nvec = D / 4;
    const int rem = D % 4;
    
    float sum = 0.0f;

    // Vectorized load with 4x unrolling
    #pragma unroll 4
    for (int i = tid; i < nvec; i += blockDim.x) {
        float4 v = __ldg(x_vec + i);
        sum += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
    }

    // Coalesced remainder handling
    const int base = nvec * 4;
    #pragma unroll 2
    for (int i = tid; i < rem; i += blockDim.x) {
        sum += fabsf(__ldg(x + row * D + base + i));
    }

    // Warp-level reduction
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane_id == 0)
        smem[threadIdx.x / warp_size] = sum;
    __syncthreads();

    // Final reduction
    float norm = 0.0f;
    if (threadIdx.x < blockDim.x / warp_size)
        norm = smem[threadIdx.x];
    
    for (int offset = blockDim.x / warp_size / 2; offset > 0; offset >>= 1)
        norm += __shfl_down_sync(0xffffffff, norm, offset);
    
    if (tid == 0)
        smem[0] = fmaxf(norm, 1e-12f);
    __syncthreads();
    
    norm = smem[0];

    // Vectorized store with 4x unrolling
    float4* out_vec = reinterpret_cast<float4*>(out + row * D);
    #pragma unroll 4
    for (int i = tid; i < nvec; i += blockDim.x) {
        float4 v = __ldg(x_vec + i);
        out_vec[i] = make_float4(v.x/norm, v.y/norm, v.z/norm, v.w/norm);
    }

    // Coalesced remainder store
    #pragma unroll 2
    for (int i = tid; i < rem; i += blockDim.x) {
        out[row * D + base + i] = __ldg(x + row * D + base + i) / norm;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
    x = x.contiguous();
    
    auto out = torch::empty_like(x);
    const int N = x.size(0);
    const int D = x.size(1);

    const int threads = (D >= 4096) ? 1024 : 
                       (D >= 2048) ? 512 :
                       (D >= 1024) ? 256 : 128;
    
    l1_norm_opt_coalesced_kernel<<<N, threads, 32*sizeof(float)>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization with optimized coalesced access");
}