#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare constant memory for logits
__constant__ float constant_logits[1024 * 100]; // Example size, adjust based on actual usage

__global__ void cross_entropy_loss_kernel_constant(
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int global_thread_id = blockIdx.x * block_size + tid;

    if (global_thread_id >= batch_size) return;

    const float* logits_i = constant_logits + global_thread_id * num_classes;
    int64_t target = targets[global_thread_id];

    // Step 1: Compute the maximum logit
    float max_logit = logits_i[0];
    for (int j = 1; j < num_classes; ++j) {
        max_logit = fmaxf(max_logit, logits_i[j]);
    }

    // Step 2: Compute the sum of exp(logits - max_logit)
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; ++j) {
        sum_exp += expf(logits_i[j] - max_logit);
    }

    // Step 3: Compute the loss
    float log_sum_exp = logf(sum_exp);
    losses[global_thread_id] = -(logits_i[target] - max_logit - log_sum_exp);
}

void copy_logits_to_constant(const float* logits, int size) {
    cudaMemcpyToSymbol(constant_logits, logits, size * sizeof(float));
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    TORCH_CHECK(targets.size(0) == batch_size, "targets must have the same batch size as predictions");

    auto losses = torch::empty({batch_size}, predictions.options());

    // Copy logits to constant memory
    copy_logits_to_constant(predictions.data_ptr<float>(), batch_size * num_classes);

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    cross_entropy_loss_kernel_constant<<<num_blocks, threads_per_block>>>(
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_constant: ", cudaGetErrorString(err));

    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss with Constant Memory Optimization (CUDA)");
}