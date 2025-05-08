#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <int BLOCK_SIZE>
__global__ void unrolled_vectorized_cosine_loss_kernel(const float* __restrict__ predictions,
                                                      const float* __restrict__ targets,
                                                      float* output,
                                                      const int N,
                                                      const int D) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int VEC_SIZE = 4;
    const int D_aligned = D & ~(VEC_SIZE-1);

    const float4* pred_vec = reinterpret_cast<const float4*>(predictions + row * D);
    const float4* target_vec = reinterpret_cast<const float4*>(targets + row * D);

    float sum_dot = 0.0f, sum_pred_sq = 0.0f, sum_target_sq = 0.0f;

    // Process vectorized elements with unrolled loop
    #pragma unroll 4
    for (int i = tid; i < D_aligned/VEC_SIZE; i += BLOCK_SIZE) {
        float4 p = pred_vec[i];
        float4 t = target_vec[i];
        sum_dot += p.x*t.x + p.y*t.y + p.z*t.z + p.w*t.w;
        sum_pred_sq += p.x*p.x + p.y*p.y + p.z*p.z + p.w*p.w;
        sum_target_sq += t.x*t.x + t.y*t.y + t.z*t.z + t.w*t.w;
    }

    // Process remaining elements
    for (int i = D_aligned + tid; i < D; i += BLOCK_SIZE) {
        float p = predictions[row*D + i];
        float t = targets[row*D + i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
    }

    __shared__ float s_dot[32], s_pred[32], s_target[32];
    if (threadIdx.x % 32 == 0) {
        s_dot[threadIdx.x/32] = sum_dot;
        s_pred[threadIdx.x/32] = sum_pred_sq;
        s_target[threadIdx.x/32] = sum_target_sq;
    }
    __syncthreads();

    if (threadIdx.x < BLOCK_SIZE/32) {
        float dot = s_dot[threadIdx.x];
        float pred = s_pred[threadIdx.x];
        float target = s_target[threadIdx.x];
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
            pred += __shfl_down_sync(0xffffffff, pred, offset);
            target += __shfl_down_sync(0xffffffff, target, offset);
        }

        if (threadIdx.x == 0) {
            const float eps = 1e-8f;
            float denom = sqrtf(pred) * sqrtf(target);
            atomicAdd(output, (1.0f - (dot / fmaxf(denom, eps))) / N);
        }
    }
}

torch::Tensor unrolled_vectorized_cosine_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have same shape");
    int N = predictions.size(0), D = predictions.size(1);
    auto output = torch::zeros({1}, predictions.options());

    const int vec_elements = 4;
    if (D >= 1024) {
        unrolled_vectorized_cosine_loss_kernel<512><<<N, 512>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), output.data_ptr<float>(), N, D);
    } else if (D >= 512) {
        unrolled_vectorized_cosine_loss_kernel<256><<<N, 256>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), output.data_ptr<float>(), N, D);
    } else {
        unrolled_vectorized_cosine_loss_kernel<128><<<N, 128>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), output.data_ptr<float>(), N, D);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &unrolled_vectorized_cosine_loss_forward, "Unrolled Vectorized Cosine Loss Forward (CUDA)");
}