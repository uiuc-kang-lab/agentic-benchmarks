#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

// Kernel that computes cross entropy loss for a chunk of the batch.
// The kernel processes samples starting from 'offset' for 'chunk_size' samples.
__global__ void cross_entropy_loss_kernel_async(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int offset,
    int chunk_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_size) return;
    int global_idx = offset + idx;

    // Get pointer to logits for sample global_idx
    const float* logits_i = logits + global_idx * num_classes;
    int64_t target = targets[global_idx];

    // Compute maximum logit for numerical stability
    float max_logit = logits_i[0];
    for (int j = 1; j < num_classes; j++) {
        max_logit = fmaxf(max_logit, logits_i[j]);
    }

    // Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++) {
        sum_exp += expf(logits_i[j] - max_logit);
    }
    float log_sum_exp = logf(sum_exp);

    // Compute cross entropy loss for the sample
    losses[global_idx] = -(logits_i[target] - max_logit - log_sum_exp);
}

// Forward function that overlaps kernel execution with async memory transfers using CUDA streams
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    // Validate input tensors
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must match batch size");

    // Allocate output tensor on GPU for per-sample losses
    auto losses_gpu = torch::empty({batch_size}, predictions.options());

    // Allocate pinned host memory for losses to allow async transfers
    auto losses_cpu = torch::empty({batch_size}, predictions.options().device(torch::kCPU)).pin_memory();

    // Define chunk size for pipelining (adjustable based on use case)
    int chunk_size = 1024;
    if (chunk_size > batch_size) chunk_size = batch_size;
    int num_chunks = (batch_size + chunk_size - 1) / chunk_size;

    // Create two CUDA streams to overlap computation with memory transfers
    cudaStream_t streams[2];
    cudaError_t err;
    for (int i = 0; i < 2; i++) {
        err = cudaStreamCreate(&streams[i]);
        TORCH_CHECK(err == cudaSuccess, "cudaStreamCreate failed: ", cudaGetErrorString(err));
    }

    const int THREADS = 256;

    // Process each chunk: launch kernel asynchronously and copy results concurrently
    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
         int offset = chunk_idx * chunk_size;
         int current_chunk_size = std::min(chunk_size, batch_size - offset);
         int blocks = (current_chunk_size + THREADS - 1) / THREADS;
         // Alternate between the two streams
         cudaStream_t stream = streams[chunk_idx % 2];

         // Launch kernel for the current chunk on the selected stream
         cross_entropy_loss_kernel_async<<<blocks, THREADS, 0, stream>>>(
              predictions.data_ptr<float>(),
              targets.data_ptr<int64_t>(),
              losses_gpu.data_ptr<float>(),
              offset,
              current_chunk_size,
              num_classes);
         err = cudaGetLastError();
         TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

         // Asynchronously copy the computed losses from GPU to pinned host memory for this chunk
         err = cudaMemcpyAsync(
              losses_cpu.data_ptr<float>() + offset,
              losses_gpu.data_ptr<float>() + offset,
              current_chunk_size * sizeof(float),
              cudaMemcpyDeviceToHost,
              stream);
         TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
    }

    // Wait for all streams to complete their tasks
    for (int i = 0; i < 2; i++) {
         err = cudaStreamSynchronize(streams[i]);
         TORCH_CHECK(err == cudaSuccess, "cudaStreamSynchronize failed: ", cudaGetErrorString(err));
         cudaStreamDestroy(streams[i]);
    }

    // Compute mean loss on CPU by accumulating the values from the pinned memory
    float sum = 0.0f;
    float* losses_ptr = losses_cpu.data_ptr<float>();
    for (int i = 0; i < batch_size; i++) {
         sum += losses_ptr[i];
    }
    float mean_loss = sum / batch_size;

    // Return the mean loss as a scalar tensor on the CPU
    return torch::tensor(mean_loss, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward with streams (CUDA)");
}
