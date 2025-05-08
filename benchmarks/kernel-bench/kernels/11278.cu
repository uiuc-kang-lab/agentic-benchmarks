#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel that computes cosine similarity loss with stride loops for correct boundary handling
__global__ void cosine_similarity_loss_kernel_strided(const float* __restrict__ predictions,
                                                       const float* __restrict__ targets,
                                                       float* output,
                                                       int N,
                                                       int D) {
    // Each block handles one row
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Initialize partial sums for dot product and squared norms
    float partial_dot = 0.0f;
    float partial_pred_sq = 0.0f;
    float partial_target_sq = 0.0f;

    // Stride loop over the D dimension to ensure all elements are processed
    for (int j = tid; j < D; j += blockDim.x) {
        float p = predictions[row * D + j];
        float t = targets[row * D + j];
        partial_dot += p * t;
        partial_pred_sq += p * p;
        partial_target_sq += t * t;
    }

    // Shared memory allocation for reduction (3 arrays each of blockDim.x floats)
    extern __shared__ float shared_mem[];
    float* s_dot = shared_mem;
    float* s_pred_sq = s_dot + blockDim.x;
    float* s_target_sq = s_pred_sq + blockDim.x;

    s_dot[tid] = partial_dot;
    s_pred_sq[tid] = partial_pred_sq;
    s_target_sq[tid] = partial_target_sq;
    __syncthreads();

    // In-block reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_dot[tid] += s_dot[tid + s];
            s_pred_sq[tid] += s_pred_sq[tid + s];
            s_target_sq[tid] += s_target_sq[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 computes the cosine similarity loss for this row and adds it atomically
    if (tid == 0) {
        const float eps = 1e-8f;
        float final_dot = s_dot[0];
        float final_pred_sq = s_pred_sq[0];
        float final_target_sq = s_target_sq[0];
        float norm_pred = sqrtf(final_pred_sq);
        float norm_target = sqrtf(final_target_sq);
        float denominator = norm_pred * norm_target;
        denominator = fmaxf(denominator, eps);
        float cos_sim = final_dot / denominator;
        float loss = 1.0f - cos_sim;
        atomicAdd(output, loss);
    }
}

// Host function launcher
torch::Tensor cosine_similarity_loss_forward_strided(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    // Allocate output tensor and initialize to zero
    auto output = torch::zeros({1}, predictions.options());
    const int block_size = 256;
    size_t shared_mem = 3 * block_size * sizeof(float);

    // Launch one block per row
    cosine_similarity_loss_kernel_strided<<<N, block_size, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D
    );

    // Average the loss across all rows
    output.div_(N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosine_similarity_loss_forward_strided, "Cosine Similarity Loss Forward with Stride Loops (CUDA)");
}
