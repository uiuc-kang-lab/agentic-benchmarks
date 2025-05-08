#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel optimized with __ldg() and 128-bit aligned loads using float4
__global__ void cosine_similarity_loss_kernel_ldg(const float* __restrict__ predictions,
                                                     const float* __restrict__ targets,
                                                     float* output,
                                                     int N,
                                                     int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Pointers to the current row
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    // Process data in 128-bit chunks (4 floats at a time)
    int numVec = D / 4;  // number of float4 loads
    int remainder = D % 4;

    // Reinterpret cast to float4 for aligned loads
    const float4* pred4 = reinterpret_cast<const float4*>(pred_row);
    const float4* target4 = reinterpret_cast<const float4*>(target_row);

    for (int i = tid; i < numVec; i += blockSize) {
        // Use __ldg() to load from read-only global memory
        float4 p4 = __ldg(&pred4[i]);
        float4 t4 = __ldg(&target4[i]);
        sum_dot    += p4.x * t4.x + p4.y * t4.y + p4.z * t4.z + p4.w * t4.w;
        sum_pred_sq   += p4.x * p4.x + p4.y * p4.y + p4.z * p4.z + p4.w * p4.w;
        sum_target_sq += t4.x * t4.x + t4.y * t4.y + t4.z * t4.z + t4.w * t4.w;
    }

    // Handle any remaining elements
    int offset = numVec * 4;
    for (int i = tid; i < remainder; i += blockSize) {
        float p = __ldg(&pred_row[offset + i]);
        float t = __ldg(&target_row[offset + i]);
        sum_dot       += p * t;
        sum_pred_sq   += p * p;
        sum_target_sq += t * t;
    }

    // Reduction in shared memory
    extern __shared__ float s_data[]; // three arrays of size blockSize
    float* s_dot = s_data;
    float* s_pred_sq = s_data + blockSize;
    float* s_target_sq = s_data + 2 * blockSize;

    s_dot[tid] = sum_dot;
    s_pred_sq[tid] = sum_pred_sq;
    s_target_sq[tid] = sum_target_sq;
    __syncthreads();

    // Parallel reduction
    for (int s = blockSize / 2; s > 0; s >>= 1) {
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
        float denom = norm_pred * norm_target;
        denom = fmaxf(denom, eps);
        float cos_sim = s_dot[0] / denom;
        float loss = 1.0f - cos_sim;
        atomicAdd(output, loss);
    }
}

// Host function launcher
torch::Tensor cosine_similarity_loss_forward_ldg(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int blockSize = 256;
    size_t shared_mem = 3 * blockSize * sizeof(float);

    // Launch one block per row
    cosine_similarity_loss_kernel_ldg<<<N, blockSize, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D
    );

    // Average the loss across rows
    output.div_(N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosine_similarity_loss_forward_ldg, "Cosine Similarity Loss Forward with __ldg optimization (CUDA)");
}
