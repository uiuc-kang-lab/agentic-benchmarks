#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void efficient_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                        const float* __restrict__ targets,
                                                        float* output,
                                                        const int N,
                                                        const int D) {
    extern __shared__ float shared_data[];

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int row = blockIdx.x;
    const int elements_per_warp = (D + blockDim.x/warpSize - 1) / (blockDim.x/warpSize);
    const int start_idx = warp_id * elements_per_warp;
    const int end_idx = min(start_idx + elements_per_warp, D);

    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float dot_sum = 0.0f;
    float pred_sq_sum = 0.0f;
    float target_sq_sum = 0.0f;

    for (int idx = start_idx + lane_id; idx < end_idx; idx += warpSize) {
        float pred = pred_row[idx];
        float target = target_row[idx];
        dot_sum += pred * target;
        pred_sq_sum += pred * pred;
        target_sq_sum += target * target;
    }

    dot_sum = warpReduceSum(dot_sum);
    pred_sq_sum = warpReduceSum(pred_sq_sum);
    target_sq_sum = warpReduceSum(target_sq_sum);

    if (lane_id == 0) {
        shared_data[warp_id] = dot_sum;
        shared_data[blockDim.x/warpSize + warp_id] = pred_sq_sum;
        shared_data[2 * (blockDim.x/warpSize) + warp_id] = target_sq_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        dot_sum = (lane_id < blockDim.x/warpSize) ? shared_data[lane_id] : 0.0f;
        pred_sq_sum = (lane_id < blockDim.x/warpSize) ? shared_data[blockDim.x/warpSize + lane_id] : 0.0f;
        target_sq_sum = (lane_id < blockDim.x/warpSize) ? shared_data[2 * (blockDim.x/warpSize) + lane_id] : 0.0f;

        dot_sum = warpReduceSum(dot_sum);
        pred_sq_sum = warpReduceSum(pred_sq_sum);
        target_sq_sum = warpReduceSum(target_sq_sum);

        if (lane_id == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(pred_sq_sum);
            float norm_target = sqrtf(target_sq_sum);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = dot_sum / denominator;
            atomicAdd(output, (1.0f - cos_sim) / N);
        }
    }
}

torch::Tensor efficient_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());

    const int threads_per_block = 256;
    size_t shared_mem = 3 * (threads_per_block / warpSize) * sizeof(float);

    efficient_cosine_similarity_loss_kernel<<<N, threads_per_block, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_cosine_similarity_loss_forward, "Efficient Cosine Similarity Loss Forward (CUDA)");
}