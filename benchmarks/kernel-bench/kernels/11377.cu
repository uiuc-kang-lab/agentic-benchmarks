#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__forceinline__ __device__ float4 load_float4_aligned(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__global__ void cosine_similarity_loss_kernel_aligned(const float* __restrict__ predictions,
                                                      const float* __restrict__ targets,
                                                      float* output,
                                                      int N,
                                                      int D) {
    extern __shared__ float shared_mem[];
    float* s_dot = shared_mem;
    float* s_pred_sq = &s_dot[blockDim.x];
    float* s_target_sq = &s_pred_sq[blockDim.x];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    // Aligned base pointers for current row
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Load data into shared memory for tiling optimization
    int tile_size = 32; // Define tile size
    for (int tile_start = 0; tile_start < D; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, D);
        if (tile_start + tid < tile_end) {
            s_pred[tile_start + tid] = __ldg(&pred_row[tile_start + tid]);
            s_target[tile_start + tid] = __ldg(&target_row[tile_start + tid]);
        }
        __syncthreads();

        // Process elements in the tile
        for (int i = tile_start + tid; i < tile_end; i += blockDim.x) {
            float pred = s_pred[i];
            float target = s_target[i];
            sum_dot += pred * target;
            sum_pred_sq += pred * pred;
            sum_target_sq += target * target;
        }
        __syncthreads();
    }
    int aligned_elements = (D / 4) * 4;
    for (int i = tid * 4; i < aligned_elements; i += blockDim.x * 4) {
        if (i + 3 < D) {
            float4 pred4 = load_float4_aligned(pred_row + i);
            float4 target4 = load_float4_aligned(target_row + i);

            // Process vector elements
            sum_dot += __ldg(&pred_row[i]) * __ldg(&target_row[i]);
            sum_dot += __ldg(&pred_row[i+1]) * __ldg(&target_row[i+1]);
            sum_dot += __ldg(&pred_row[i+2]) * __ldg(&target_row[i+2]);
            sum_dot += __ldg(&pred_row[i+3]) * __ldg(&target_row[i+3]);

            sum_pred_sq += pred4.x * pred4.x + pred4.y * pred4.y + 
                          pred4.z * pred4.z + pred4.w * pred4.w;
            sum_target_sq += target4.x * target4.x + target4.y * target4.y + 
                            target4.z * target4.z + target4.w * target4.w;
        }
    }

    // Handle remaining elements
    for (int i = aligned_elements + tid; i < D; i += blockDim.x) {
        float pred = __ldg(&pred_row[i]);
        float target = __ldg(&target_row[i]);
        sum_dot += pred * target;
        sum_pred_sq += pred * pred;
        sum_target_sq += target * target;
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
    }

    // First thread in each warp writes to shared memory
    if (lane == 0) {
        s_dot[warp_id] = sum_dot;
        s_pred_sq[warp_id] = sum_pred_sq;
        s_target_sq[warp_id] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid == 0) {
        sum_dot = 0.0f;
        sum_pred_sq = 0.0f;
        sum_target_sq = 0.0f;

        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < num_warps; i++) {
            sum_dot += s_dot[i];
            sum_pred_sq += s_pred_sq[i];
            sum_target_sq += s_target_sq[i];
        }

        const float eps = 1e-8f;
        float norm_pred = sqrtf(sum_pred_sq);
        float norm_target = sqrtf(sum_target_sq);
        float denominator = norm_pred * norm_target;
        denominator = fmaxf(denominator, eps);
        float cos_sim = sum_dot / denominator;
        atomicAdd(output, 1.0f - cos_sim);
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
    
    // Ensure block size is multiple of warp size for optimal performance
    const int block_size = 256;
    const int num_warps = (block_size + 31) / 32;
    size_t shared_mem = 3 * block_size * sizeof(float);

    cosine_similarity_loss_kernel_aligned<<<N, block_size, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    output.div_(N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward with aligned memory access (CUDA)");
}