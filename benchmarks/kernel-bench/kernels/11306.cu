#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle instructions
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void shared_memory_optimized_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                                       const float* __restrict__ targets,
                                                                       float* output,
                                                                       int N,
                                                                       int D) {
    extern __shared__ float shared[];
    float* s_dot = shared;
    float* s_pred_sq = shared + blockDim.x;
    float* s_target_sq = shared + 2 * blockDim.x;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;

    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Vectorized load parameters
    const int vec_size = 4;
    int D_aligned = (D / vec_size) * vec_size;

    for (int i = tid; i < D_aligned; i += blockDim.x) {
        float4 p = reinterpret_cast<const float4*>(pred_row)[i / vec_size];
        float4 t = reinterpret_cast<const float4*>(target_row)[i / vec_size];

        sum_dot += p.x * t.x + p.y * t.y + p.z * t.z + p.w * t.w;
        sum_pred_sq += p.x * p.x + p.y * p.y + p.z * p.z + p.w * p.w;
        sum_target_sq += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
    }

    for (int i = D_aligned + tid; i < D; i += blockDim.x) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    s_dot[tid] = sum_dot;
    s_pred_sq[tid] = sum_pred_sq;
    s_target_sq[tid] = sum_target_sq;
    __syncthreads();

    // Perform reduction in shared memory
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
        atomicAdd(output, (1.0f - cos_sim) / N);
    }
}

// Host function binding the CUDA kernel to PyTorch

torch::Tensor shared_memory_optimized_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 512;
    size_t shared_mem_size = 3 * block_size * sizeof(float);

    shared_memory_optimized_cosine_similarity_loss_kernel<<<N, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_optimized_cosine_similarity_loss_forward, "Shared Memory Optimized Cosine Similarity Loss Forward (CUDA)");
}