#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 512

__global__ void vectorized_cosine_loss_kernel(const float* __restrict__ predictions,
                                               const float* __restrict__ targets,
                                               float* output,
                                               int N,
                                               int D) {
    extern __shared__ float s_tile[];
    float* pred_tile = s_tile;
    float* target_tile = s_tile + TILE_SIZE;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Vectorized load helper
    auto load_vector = [](const float* ptr) -> float4 {
        if (reinterpret_cast<uintptr_t>(ptr) % 16 == 0) {
            return *reinterpret_cast<const float4*>(ptr);
        }
        return {0}; // Shouldn't hit with aligned tensors
    };

    for (int i = 0; i < D; i += TILE_SIZE) {
        int tile_end = min(i + TILE_SIZE, D);
        int chunk_size = tile_end - i;

        // Vectorized tile loading with coalesced access
        for (int j = 0; j < chunk_size; j += 4 * stride) {
            int idx = i + tid * 4 + j;
            if (idx + 3 < tile_end) {
                float4 p_vec = load_vector(&pred_row[idx]);
                float4 t_vec = load_vector(&target_row[idx]);
                pred_tile[tid*4] = p_vec.x; pred_tile[tid*4+1] = p_vec.y;
                pred_tile[tid*4+2] = p_vec.z; pred_tile[tid*4+3] = p_vec.w;
                target_tile[tid*4] = t_vec.x; target_tile[tid*4+1] = t_vec.y;
                target_tile[tid*4+2] = t_vec.z; target_tile[tid*4+3] = t_vec.w;
            }
        }
        __syncthreads();

        // Process tile with manual loop unrolling
        for (int j = 0; j < TILE_SIZE; j += 4) {
            float p = pred_tile[tid + j];
            float t = target_tile[tid + j];
            sum_dot += p * t;
            sum_pred_sq += p * p;
            sum_target_sq += t * t;
            if (tid + j + 1 >= TILE_SIZE) break;
        }
        __syncthreads();
    }

    // Warp reduction using shuffle primitives
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
    }

    // Store partial sum and  warp reduction
    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, 1.0f - (sum_dot / (sqrtf(sum_pred_sq) * sqrtf(sum_target_sq) + 1e-8f)));
    }
}

torch::Tensor cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have same shape");
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "Inputs must be CUDA tensors");

    auto output = torch::zeros({1}, predictions.options());
    int block_size = 256;
    int shared_mem = 2 * TILE_SIZE * sizeof(float);

    vectorized_cosine_loss_kernel<<<predictions.size(0), block_size, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        predictions.size(0),
        predictions.size(1)
    );

    output.div_(predictions.size(0));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosine_similarity_loss_forward, "Vectorized cosine loss forward");
}
