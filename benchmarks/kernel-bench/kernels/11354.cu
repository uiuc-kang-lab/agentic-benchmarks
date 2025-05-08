#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Struct to hold partial sums for cosine similarity computation
struct Sums {
    float dot;
    float pred_sq;
    float target_sq;
};

// Device function to compute partial sums of dot product and squared norms
__device__ Sums compute_partial_sums(const float* __restrict__ pred_row,
                                     const float* __restrict__ target_row,
                                     int D,
                                     int tid,
                                     int blockDim) {
    Sums sums = {0.0f, 0.0f, 0.0f};
    for (int i = tid; i < D; i += blockDim) {
        float p = pred_row[i];
        float t = target_row[i];
        sums.dot += p * t;
        sums.pred_sq += p * p;
        sums.target_sq += t * t;
    }
    return sums;
}

// Device function to perform shared memory reduction for the three arrays
__device__ void reduce_shared_sums(float* s_dot,
                                   float* s_pred_sq,
                                   float* s_target_sq,
                                   int blockDim) {
    for (int s = blockDim / 2; s > 0; s >>= 1) {
        int tid = threadIdx.x;
        if (tid < s) {
            s_dot[tid] += s_dot[tid + s];
            s_pred_sq[tid] += s_pred_sq[tid + s];
            s_target_sq[tid] += s_target_sq[tid + s];
        }
        __syncthreads();
    }
}

// Device function to compute the cosine similarity loss from the reduced sums
__device__ float compute_loss_from_sums(const Sums &sums) {
    const float eps = 1e-8f;
    float norm_pred = sqrtf(sums.pred_sq);
    float norm_target = sqrtf(sums.target_sq);
    float denominator = norm_pred * norm_target;
    denominator = fmaxf(denominator, eps);
    float cos_sim = sums.dot / denominator;
    return 1.0f - cos_sim;
}

// Kernel function: Each block computes the loss for one row
__global__ void modular_cosine_similarity_loss_kernel(const float* __restrict__ predictions,
                                                       const float* __restrict__ targets,
                                                       float* output,
                                                       int N,
                                                       int D) {
    extern __shared__ float s_data[];
    float* s_dot = s_data;
    float* s_pred_sq = s_data + blockDim.x;
    float* s_target_sq = s_pred_sq + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* pred_row = predictions + row * D;
    const float* target_row = targets + row * D;

    // Compute partial sums in a modular fashion
    Sums my_sums = compute_partial_sums(pred_row, target_row, D, tid, blockDim.x);

    // Store local partial results in shared memory
    s_dot[tid] = my_sums.dot;
    s_pred_sq[tid] = my_sums.pred_sq;
    s_target_sq[tid] = my_sums.target_sq;
    __syncthreads();

    // Perform reduction on the partial sums using a modular device function
    reduce_shared_sums(s_dot, s_pred_sq, s_target_sq, blockDim.x);

    if (tid == 0) {
        Sums total = { s_dot[0], s_pred_sq[0], s_target_sq[0] };
        float loss = compute_loss_from_sums(total);
        atomicAdd(output, loss);
    }
}

// Host function to launch the kernel
torch::Tensor modular_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int block_size = 256;
    size_t shared_mem = 3 * block_size * sizeof(float);

    modular_cosine_similarity_loss_kernel<<<N, block_size, shared_mem>>>(
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
    m.def("forward", &modular_cosine_similarity_loss_forward, "Modular Cosine Similarity Loss Forward (CUDA)");
}
