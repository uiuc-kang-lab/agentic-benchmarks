#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr int VEC_SIZE = 4;

template<typename T>
struct VectorType { using Type = T; };
template<>
struct VectorType<float> { using Type = float4; };

__device__ __forceinline__ void warp_reduce(float& sum_dot, float& sum_pred_sq, float& sum_target_sq) {
    for(int offset = warpSize/2; offset >= 1; offset /= 2) {
        sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
    }
}

__global__ void cosine_similarity_loss_kernel(const float* __restrict__ pred,
                                             const float* __restrict__ targ,
                                             float* output,
                                             int N,
                                             int D) {
    extern __shared__ __align__(sizeof(float4)) char sdata_[];
    using vec_t = VectorType<float>::Type;

    const int vec_d = D / VEC_SIZE;
    const int thread_stride = blockDim.x * VEC_SIZE;
    
    float* spred = reinterpret_cast<float*>(sdata_);
    float* starg = spred + D;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Vectorized shared memory loading
    const float* row_pred = pred + row * D;
    const float* row_targ = targ + row * D;

    for(int i = tid * VEC_SIZE; i < D; i += thread_stride) {
        if(i + VEC_SIZE <= D) {
            reinterpret_cast<vec_t*>(spred)[i/VEC_SIZE] = 
                reinterpret_cast<const vec_t*>(row_pred)[i/VEC_SIZE];
            reinterpret_cast<vec_t*>(starg)[i/VEC_SIZE] = 
                reinterpret_cast<const vec_t*>(row_targ)[i/VEC_SIZE];
        } else {
            for(int j = 0; j < VEC_SIZE && (i + j) < D; ++j) {
                spred[i + j] = row_pred[i + j];
                starg[i + j] = row_targ[i + j];
            }
        }
    }
    __syncthreads();

    // Partial sum accumulators
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Compute with unrolled vector access
    for(int i = tid * VEC_SIZE; i < D; i += thread_stride) {
        float p[VEC_SIZE], t[VEC_SIZE];
        for(int j = 0; j < VEC_SIZE; ++j) {
            if(i + j < D) {
                p[j] = spred[i + j];
                t[j] = starg[i + j];
            } else {
                p[j] = 0.f;
                t[j] = 0.f;
            }
        }

        for(int j = 0; j < VEC_SIZE; ++j) {
            sum_dot += p[j] * t[j];
            sum_pred_sq += p[j] * p[j];
            sum_target_sq += t[j] * t[j];
        }
    }

    // Final warp reduction
    warp_reduce(sum_dot, sum_pred_sq, sum_target_sq);

    // Save block result
    if(threadIdx.x == 0) {
        const float eps = 1e-8f;
        float denom = sqrtf(sum_pred_sq) * sqrtf(sum_target_sq) + eps;
        atomicAdd(output, 1.0f - (sum_dot / denom));
    }
}

torch::Tensor cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int max_block = 128;
    const int threads = min(max_block, (D + VEC_SIZE - 1) / VEC_SIZE);
    const size_t smem = 2 * D * sizeof(float);

    cosine_similarity_loss_kernel<<<N, threads, smem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D
    );

    output.div_(N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Loss Optimized Reductions (CUDA)");
}