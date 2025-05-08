#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <math.h>

// CUDA kernel to compute cross entropy loss for a chunk of data.
// The pointers are assumed to be offset to the beginning of the chunk.
__global__ void cross_entropy_loss_kernel_chunk(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int chunk_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    for (int i = idx; i < chunk_size; i += total_threads) {
        const float* sample_logits = logits + i * num_classes;
        int target = targets[i];

        // Compute max logit
        float max_logit = sample_logits[0];
        for (int j = 1; j < num_classes; j++) {
            max_logit = fmaxf(max_logit, sample_logits[j]);
        }

        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(sample_logits[j] - max_logit);
        }

        float log_sum_exp = logf(sum_exp);
        losses[i] = -(sample_logits[target] - max_logit - log_sum_exp);
    }
}


// The forward function splits the batch into chunks and uses multiple CUDA streams
// to overlap the kernel computations with asynchronous device-to-host memory transfers.
// The individual losses for each sample are copied to pinned host memory and then reduced on the CPU.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be a Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be an Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have the same batch size as predictions");

    // Allocate a device tensor for losses
    auto losses_device = torch::empty({batch_size}, predictions.options());

    // Allocate pinned host memory for asynchronous copy
    auto losses_host = torch::empty({batch_size}, torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(torch::kCPU)
                                      .pin_memory(true));

    // Determine number of streams (pipelining) - choose 4 or adjust based on batch size
    int num_streams = 4;
    if(batch_size < num_streams) num_streams = batch_size;
    int chunk_size = (batch_size + num_streams - 1) / num_streams;

    std::vector<cudaStream_t> streams(num_streams);
    for (int s = 0; s < num_streams; s++) {
        cudaStreamCreate(&streams[s]);
    }

    int threads = 256;

    // For each chunk, launch the kernel on its own stream and copy the results asynchronously to host.
    for (int s = 0; s < num_streams; s++) {
        int start = s * chunk_size;
        int end = std::min(start + chunk_size, batch_size);
        int current_chunk = end - start;
        if (current_chunk <= 0)
            break;

        int blocks = (current_chunk + threads - 1) / threads;
        
        // Launch the kernel for this chunk. Adjust pointers by offset.
        cross_entropy_loss_kernel_chunk<<<blocks, threads, 0, streams[s]>>>(
            predictions.data_ptr<float>() + start * num_classes,
            targets.data_ptr<int64_t>() + start,
            losses_device.data_ptr<float>() + start,
            current_chunk,
            num_classes
        );

        // Asynchronously copy the computed losses for this chunk from device to pinned host memory
        cudaMemcpyAsync(losses_host.data_ptr<float>() + start,
                        losses_device.data_ptr<float>() + start,
                        current_chunk * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        streams[s]);
    }

    // Synchronize all streams
    for (int s = 0; s < num_streams; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    // Perform reduction on the host (using the pinned memory) to compute the mean loss.
    float sum = 0.0f;
    float* losses_ptr = losses_host.data_ptr<float>();
    for (int i = 0; i < batch_size; i++) {
        sum += losses_ptr[i];
    }
    float mean_loss = sum / batch_size;

    // Return the mean loss as a tensor on the same device as the input predictions
    auto result = torch::tensor(mean_loss, predictions.options());
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward with overlapped computation and memory transfers (CUDA)");
}
