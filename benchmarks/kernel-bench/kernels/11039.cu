#include <torch/extension.h>
#include <vector>

#define NUM_STREAMS 4
#define CHUNK_SIZE 64

__global__ void cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes,
    int offset
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch_size)
    {
        int global_idx = tid + offset;
        const float* logits_i = logits + tid * num_classes;
        int64_t target = targets[tid];

        float max_logit = logits_i[0];
        #pragma unroll 16
        for (int j = 1; j < num_classes; j++)
        {
            if (logits_i[j] > max_logit)
                max_logit = logits_i[j];
        }

        float sum_exp = 0.0f;
        #pragma unroll 16
        for (int j = 0; j < num_classes; j++)
        {
            sum_exp += expf(logits_i[j] - max_logit);
        }

        float log_sum_exp = logf(sum_exp);
        float loss = -(logits_i[target] - max_logit - log_sum_exp);
        losses[global_idx] = loss;
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets)
{
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    auto losses = torch::empty({batch_size}, predictions.options());

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int threads = 256;
    
    // Process data in chunks using multiple streams
    for (int chunk_start = 0; chunk_start < batch_size; chunk_start += CHUNK_SIZE) {
        int current_chunk_size = std::min(CHUNK_SIZE, batch_size - chunk_start);
        int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
        int blocks = (current_chunk_size + threads - 1) / threads;

        // Offset pointers for this chunk
        const float* chunk_logits = predictions.data_ptr<float>() + chunk_start * num_classes;
        const int64_t* chunk_targets = targets.data_ptr<int64_t>() + chunk_start;
        
        cross_entropy_loss_kernel<<<blocks, threads, 0, streams[stream_idx]>>>(
            chunk_logits,
            chunk_targets,
            losses.data_ptr<float>(),
            current_chunk_size,
            num_classes,
            chunk_start
        );
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in cross_entropy_loss_kernel: ", cudaGetErrorString(err));

    // Compute mean loss over batch
    auto loss = losses.mean();
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Pipelined Cross Entropy Loss forward (CUDA)");
}