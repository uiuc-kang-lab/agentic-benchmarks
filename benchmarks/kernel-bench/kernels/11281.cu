#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void balanced_work_distribution_kernel(const float* __restrict__ predictions,
                                                   const float* __restrict__ targets,
                                                   float* output,
                                                   int N,
                                                   int D) {
    extern __shared__ float s_data[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_width = gridDim.x * blockDim.x;
    int stride = blockDim.x;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    for (int i = tid + blockIdx.x * stride; i < N * D; i += block_width) {
        float p = predictions[i];
        float t = targets[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    float* s_dot = s_data;
    float* s_pred_sq = s_data + blockDim.x;
    float* s_target_sq = s_pred_sq + blockDim.x;

    s_dot[tid] = sum_dot;
    s_pred_sq[tid] = sum_pred_sq;
    s_target_sq[tid] = sum_target_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_dot[tid] += s_dot[tid + s];
            s_pred_sq[tid] += s_pred_sq[tid + s];
            s_target_sq[tid] += s_target_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float eps = 1e-8f;
        float norm_pred = sqrtf(s_pred_sq[0]);
        float norm_target = sqrtf(s_target_sq[0]);
        float denominator = norm_pred * norm_target;
        denominator = fmaxf(denominator, eps);
        
        float cos_sim = s_dot[0] / denominator;
        atomicAdd(output, (1.0f - cos_sim) / N); // Accumulate contribution for this row
    }
}

torch::Tensor balanced_work_distribution_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int block_size = 256;
    int num_blocks = (N * D + block_size - 1) / block_size;
    size_t shared_mem = 3 * block_size * sizeof(float);

    balanced_work_distribution_kernel<<<num_blocks, block_size, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &balanced_work_distribution_forward, "Balanced Work Distribution Cosine Similarity Loss Forward (CUDA)");}
