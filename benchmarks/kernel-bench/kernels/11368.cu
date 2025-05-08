#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NUM_STREAMS 4
#define CHUNK_SIZE 32

__global__ void cosine_similarity_loss_kernel_streamed(const float* __restrict__ predictions,
                                                     const float* __restrict__ targets,
                                                     float* output,
                                                     int N,
                                                     int D,
                                                     int offset) {
    int row = blockIdx.x + offset;
    if (row >= N) return;
    
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warpId = tid >> 5;
    
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Coalesced memory access pattern
    for (int i = tid; i < D; i += blockDim.x) {
        float p = pred_row[i];
        float t = target_row[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
    }

    int numWarps = (blockDim.x + 31) / 32;
    extern __shared__ float shared[];
    float* s_dot = shared;
    float* s_pred_sq = s_dot + numWarps;
    float* s_target_sq = s_pred_sq + numWarps;

    if (lane == 0) {
        s_dot[warpId] = sum_dot;
        s_pred_sq[warpId] = sum_pred_sq;
        s_target_sq[warpId] = sum_target_sq;
    }
    __syncthreads();

    if (tid < numWarps) {
        sum_dot = s_dot[tid];
        sum_pred_sq = s_pred_sq[tid];
        sum_target_sq = s_target_sq[tid];
        
        #pragma unroll
        for (int offset = (numWarps >> 1); offset > 0; offset >>= 1) {
            sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
            sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
            sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
        }

        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(sum_pred_sq);
            float norm_target = sqrtf(sum_target_sq);
            float denominator = fmaxf(norm_pred * norm_target, eps);
            float cos_sim = sum_dot / denominator;
            atomicAdd(output, 1.0f - cos_sim);
        }
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
    const int block_size = 256; // Ensure block size is a multiple of 32
    int numWarps = (block_size + 31) / 32;
    size_t shared_mem = 3 * numWarps * sizeof(float);

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Process data in chunks using multiple streams
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE * NUM_STREAMS) {
        for (int s = 0; s < NUM_STREAMS && (chunk_start + s * CHUNK_SIZE) < N; s++) {
            int chunk_size = std::min(CHUNK_SIZE, N - (chunk_start + s * CHUNK_SIZE));
            
            cosine_similarity_loss_kernel_streamed<<<chunk_size, block_size, shared_mem, streams[s]>>>(
                predictions.data_ptr<float>(),
                targets.data_ptr<float>(),
                output.data_ptr<float>(),
                N,
                D,
                chunk_start + s * CHUNK_SIZE
            );
        }
    }

    // Synchronize and cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    output.div_(N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward with streams (CUDA)");
}