#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void l1_norm_streams_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      int N, int D,
                                      int row_start) {
    int row = row_start + blockIdx.x;
    if (row >= N) return;
    
    int row_offset = row * D;
    float local_sum = 0.0f;
    int nvec = D / 4;
    int rem = D % 4;

    const float4* x_vec = reinterpret_cast<const float4*>(x + row_offset);
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float4 v = __ldg(x_vec + i);
        local_sum += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
    }

    int base = nvec * 4;
    for (int j = threadIdx.x; j < rem; j += blockDim.x) {
        local_sum += fabsf(x[row_offset + base + j]);
    }

    float warp_sum = warpReduceSum(local_sum);
    extern __shared__ float shared[];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5; 
    if (lane == 0) shared[warp_id] = warp_sum;
    __syncthreads();

    float block_sum = (threadIdx.x < (blockDim.x + 31)/32) ? shared[threadIdx.x] : 0;
    block_sum = warpReduceSum(block_sum);

    if (threadIdx.x == 0) shared[0] = fmaxf(block_sum, 1e-12f);
    __syncthreads();
    float norm = shared[0];

    float4* out_vec = reinterpret_cast<float4*>(out + row_offset);
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float4 v = __ldg(x_vec + i);
        out_vec[i] = {v.x/norm, v.y/norm, v.z/norm, v.w/norm};
    }
    for (int j = threadIdx.x; j < rem; j += blockDim.x) {
        out[row_offset + base + j] = x[row_offset + base + j] / norm;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    x = x.contiguous();
    
    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);

    int threads = 256;
    if (D >= 1024) threads = 1024;
    else if (D >= 512) threads = 512;
    threads = std::min(threads, D);
    threads = ((threads + 31)/32)*32;
    int shared_mem = (threads/32) * sizeof(float);

    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i=0; i<num_streams; ++i)
        cudaStreamCreate(&streams[i]);

    int chunk = (N + num_streams - 1) / num_streams;
    for (int i=0; i<num_streams; ++i) {
        int start = i*chunk;
        int count = std::min(chunk, N-start);
        if (count <= 0) continue;
        
        l1_norm_streams_kernel<<<count, threads, shared_mem, streams[i]>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            N, D, start
        );
    }

    for (int i=0; i<num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization with stream-parallel execution");
}