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

__global__ void l1_norm_stride_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int N,
                                       int D) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_size = 32;
    const int lane_id = tid % warp_size;
    const int warp_id = tid / warp_size;
    
    float local_sum = 0.0f;
    for (int col = tid; col < D; col += blockDim.x) {
        local_sum += fabsf(x[row * D + col]);
    }

    float warp_sum = warpReduceSum(local_sum);

    if (lane_id == 0) {
        sdata[warp_id] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (tid < blockDim.x / warp_size) ? sdata[tid] : 0.0f;
        block_sum = warpReduceSum(block_sum);
        if (tid == 0) {
            sdata[0] = fmaxf(block_sum, 1e-12f);
        }
    }
    __syncthreads();
    float norm = sdata[0];

    for (int col = tid; col < D; col += blockDim.x) {
        out[row * D + col] = x[row * D + col] / norm;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    const int N = x.size(0);
    const int D = x.size(1);

    const int warp_size = 32;
    const int threads = std::min<int>(1024, D);
    const int shared_mem = (threads / warp_size) * sizeof(float);

    l1_norm_stride_kernel<<<N, threads, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization with stride loops (CUDA)");
}
