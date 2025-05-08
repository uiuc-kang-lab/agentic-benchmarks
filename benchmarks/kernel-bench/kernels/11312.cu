#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__inline__ __device__ float warp_reduce(float val) {
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void optimized_cosine_loss_kernel(const float* __restrict__ pred,
                                             const float* __restrict__ target,
                                             float* output,
                                             int N,
                                             int D) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int vec_size = 4;
    const int D_vec = D / vec_size;

    const float4* pred_vec = reinterpret_cast<const float4*>(pred + row*D);
    const float4* target_vec = reinterpret_cast<const float4*>(target + row*D);

    float dot = 0.0f, p_sq = 0.0f, t_sq = 0.0f;

    // Vectorized load with coalesced access
    #pragma unroll
    for (int i = tid; i < D_vec; i += blockDim.x) {
        float4 p = pred_vec[i];
        float4 t = target_vec[i];
        dot += p.x*t.x + p.y*t.y + p.z*t.z + p.w*t.w;
        p_sq += p.x*p.x + p.y*p.y + p.z*p.z + p.w*p.w;
        t_sq += t.x*t.x + t.y*t.y + t.z*t.z + t.w*t.w;
    }

    // Handle remainder elements
    for (int i = D_vec*vec_size + tid; i < D; i += blockDim.x) {
        float p = pred[row*D + i];
        float t = target[row*D + i];
        dot += p * t;
        p_sq += p * p;
        t_sq += t * t;
    }

    // Warp-level reduction
    dot = warp_reduce(dot);
    p_sq = warp_reduce(p_sq);
    t_sq = warp_reduce(t_sq);

    // Shared memory for cross-warp reduction
    __shared__ float smem[3][32];
    if (threadIdx.x % warpSize == 0) {
        smem[0][threadIdx.x/warpSize] = dot;
        smem[1][threadIdx.x/warpSize] = p_sq;
        smem[2][threadIdx.x/warpSize] = t_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    if (threadIdx.x < 32) {
        dot = threadIdx.x < blockDim.x/warpSize ? smem[0][threadIdx.x] : 0;
        p_sq = threadIdx.x < blockDim.x/warpSize ? smem[1][threadIdx.x] : 0;
        t_sq = threadIdx.x < blockDim.x/warpSize ? smem[2][threadIdx.x] : 0;

        dot = warp_reduce(dot);
        p_sq = warp_reduce(p_sq);
        t_sq = warp_reduce(t_sq);

        if (threadIdx.x == 0) {
            const float eps = 1e-8f;
            float denom = sqrtf(p_sq) * sqrtf(t_sq);
            atomicAdd(output, (1.0f - (dot / fmaxf(denom, eps))) / N);
        }
    }
}

torch::Tensor optimized_cosine_loss_forward(torch::Tensor pred, torch::Tensor target) {
    TORCH_CHECK(pred.dim() == 2 && target.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(pred.sizes() == target.sizes(), "Shape mismatch");

    auto output = torch::zeros({1}, pred.options());
    const int block_size = 512;
    optimized_cosine_loss_kernel<<<pred.size(0), block_size>>>(
        pred.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        pred.size(0),
        pred.size(1)
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_cosine_loss_forward, "Optimized Cosine Loss Forward (CUDA)");
}