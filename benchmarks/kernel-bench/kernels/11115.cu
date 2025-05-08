#include <torch/extension.h>

__global__ void cross_entropy_loss_kernel(
    const float* logits,
    const int64_t* targets,
    float* losses,
    int start_idx,
    int chunk_size,
    int num_classes
)
{
    int i = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < start_idx + chunk_size)
    {
        const float* logits_i = logits + i * num_classes;
        int64_t target = targets[i];

        float max_logit = logits_i[0];
        for (int j = 1; j < num_classes; j++)
            if (logits_i[j] > max_logit) max_logit = logits_i[j];

        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++)
            sum_exp += expf(logits_i[j] - max_logit);

        losses[i] = -(logits_i[target] - max_logit - logf(sum_exp));
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets)
{
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(predictions.dim() == 2 && targets.dim() == 1, "Invalid input dimensions");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    auto losses = torch::empty({batch_size}, predictions.options());

    const int threads = 256;
    const int num_streams = 4;
    int chunk_size = (batch_size + num_streams - 1) / num_streams;

    std::vector<torch::cuda::Stream> streams(num_streams);
    for (int s = 0; s < num_streams; ++s) {
        streams[s] = torch::cuda::Stream(torch::cuda::current_device());
        int start = s * chunk_size;
        int current_chunk = std::min(chunk_size, batch_size - start);
        if (current_chunk <= 0) continue;

        int blocks = (current_chunk + threads - 1) / threads;
        cudaStream_t stream_handle = streams[s].stream();

        cross_entropy_loss_kernel<<<blocks, threads, 0, stream_handle>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<int64_t>(),
            losses.data_ptr<float>(),
            start,
            current_chunk,
            num_classes
        );
    }

    for (auto& stream : streams)
        stream.synchronize();

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed Cross Entropy Loss");
}