#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cross_entropy_loss_kernel_streamed(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    extern __shared__ float shared_logits[];

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_id >= batch_size) return;

    const float* sample_logits = logits + global_thread_id * num_classes;
    int target_class = targets[global_thread_id];

    // Load logits into shared memory
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        shared_logits[j] = sample_logits[j];
    }
    __syncthreads();

    // Compute the maximum logit
    float max_logit = -1e38f;
    for (int j = 0; j < num_classes; ++j) {
        max_logit = fmaxf(max_logit, shared_logits[j]);
    }

    // Compute the sum of exp(logits - max_logit)
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; ++j) {
        sum_exp += expf(shared_logits[j] - max_logit);
    }

    // Compute the loss for the sample
    float loss = -(shared_logits[target_class] - max_logit - logf(sum_exp));
    losses[global_thread_id] = loss;
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

    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    size_t shared_memory_size = num_classes * sizeof(float);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cross_entropy_loss_kernel_streamed<<<blocks, threads_per_block, shared_memory_size, stream>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel_streamed: ", cudaGetErrorString(err));

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Cross Entropy Loss with CUDA Streams");
}