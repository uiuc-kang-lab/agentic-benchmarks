#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void workload_balanced_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                                 const float* __restrict__ targets,
                                                                 float* output,
                                                                 const int N,
                                                                 const int D) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int total_threads = blockDim.x * gridDim.x;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    for (int i = tid; i < N * D; i += total_threads) {
        float p = predictions[i];
        float t = targets[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    sum_dot = warp_reduce_sum(sum_dot);
    sum_pred_sq = warp_reduce_sum(sum_pred_sq);
    sum_target_sq = warp_reduce_sum(sum_target_sq);

    __shared__ float s_dot[32];
    __shared__ float s_pred_sq[32];
    __shared__ float s_target_sq[32];

    if (threadIdx.x % warpSize == 0) {
        s_dot[threadIdx.x / warpSize] = sum_dot;
        s_pred_sq[threadIdx.x / warpSize] = sum_pred_sq;
        s_target_sq[threadIdx.x / warpSize] = sum_target_sq;
    }
    __syncthreads();

    if (threadIdx.x < warpSize) {
        int num_warps = blockDim.x / warpSize;
        sum_dot = threadIdx.x < num_warps ? s_dot[threadIdx.x] : 0.0f;
        sum_pred_sq = threadIdx.x < num_warps ? s_pred_sq[threadIdx.x] : 0.0f;
        sum_target_sq = threadIdx.x < num_warps ? s_target_sq[threadIdx.x] : 0.0f;

        sum_dot = warp_reduce_sum(sum_dot);
        sum_pred_sq = warp_reduce_sum(sum_pred_sq);
        sum_target_sq = warp_reduce_sum(sum_target_sq);

        if (threadIdx.x == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(sum_pred_sq);
            float norm_target = sqrtf(sum_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = sum_dot / denominator;
            atomicAdd(output, (1.0f - cos_sim) / N);
        }
    }
}

torch::Tensor workload_balanced_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int block_size = 256;
    const int grid_size = min((N * D + block_size - 1) / block_size, 1024);

    workload_balanced_cosine_similarity_loss_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &workload_balanced_cosine_similarity_loss_forward, "Workload Balanced Cosine Similarity Loss Forward (CUDA)");
}