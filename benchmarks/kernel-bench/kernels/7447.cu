#include <torch/extension.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Ensure inputs are on CUDA and contiguous
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Get batch size and calculate chunk size
    int64_t batch_size = x.size(0);
    int64_t chunk_size = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;

    // Create output tensor
    // Calculate output size manually for conv transpose 2d
    auto input_size = x.sizes();
    auto weight_size = weight.sizes();
    int64_t N = input_size[0];
    int64_t H = (input_size[2] - 1) * stride - 2 * padding + weight_size[2] + output_padding;
    int64_t W = (input_size[3] - 1) * stride - 2 * padding + weight_size[3] + output_padding;
    int64_t C_out = weight_size[1] * groups;
    auto output_size = torch::IntArrayRef({N, C_out, H, W});
    auto output = torch::empty(output_size, x.options());

    // Process chunks in parallel streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int64_t start_idx = i * chunk_size;
        int64_t end_idx = std::min(start_idx + chunk_size, batch_size);

        if (start_idx >= batch_size) break;

        // Create view of input chunk
        auto x_chunk = x.narrow(0, start_idx, end_idx - start_idx);
        auto output_chunk = output.narrow(0, start_idx, end_idx - start_idx);

        // Process chunk using current stream
        at::cuda::CUDAStreamGuard stream_guard(streams[i]);
        output_chunk.copy_(
            at::conv_transpose2d(
                x_chunk,
                weight,
                bias,
                {stride, stride},
                {padding, padding},
                {output_padding, output_padding},
                groups
            )
        );
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}