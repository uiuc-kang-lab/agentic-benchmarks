#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Threshold constants for switching between implementations
constexpr int BATCH_SIZE_THRESHOLD = 16;
constexpr int CHANNEL_THRESHOLD = 64;

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    // Get input dimensions
    auto x_sizes = x.sizes();
    int N = x_sizes[0];
    int C_in = x_sizes[1];
    
    // Decision logic for which implementation to use
    bool use_custom = (N <= BATCH_SIZE_THRESHOLD) && (C_in <= CHANNEL_THRESHOLD);
    
    if (use_custom) {
        // Use custom kernel implementation for small batches/channels
        int H = x_sizes[2];
        int W = x_sizes[3];
        auto w_sizes = weight.sizes();
        int C_out = w_sizes[1];
        int K = w_sizes[2];
        int H_out = (H - 1) * stride - 2 * padding + K + output_padding;
        int W_out = (W - 1) * stride - 2 * padding + K + output_padding;

        auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());
        
        // Use 2 streams for overlapping computation
        cudaStream_t streams[2];
        for (int i = 0; i < 2; i++) {
            cudaStreamCreate(&streams[i]);
        }

        int chunk = (N + 1) / 2;
        int block_size = 256;
        const float* weight_ptr = weight.data_ptr<float>();

        // Launch kernels on separate streams
        for (int i = 0; i < 2; i++) {
            int start = i * chunk;
            int end = std::min(N, (i + 1) * chunk);
            int chunkN = end - start;
            if (chunkN <= 0) continue;

            int num_elements = chunkN * C_in * H * W;
            const float* x_ptr = x.data_ptr<float>() + start * C_in * H * W;
            float* out_ptr = output.data_ptr<float>() + start * C_out * H_out * W_out;

            int grid_size = (num_elements + block_size - 1) / block_size;

            conv_transpose2d_kernel<<<grid_size, block_size, 0, streams[i]>>>(
                x_ptr, weight_ptr, out_ptr,
                chunkN, C_in, H, W,
                C_out, K, stride, padding,
                H_out, W_out
            );
        }

        // Cleanup streams
        for (int i = 0; i < 2; i++) {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }

        if (bias.has_value()) {
            // Add bias using vectorized operations
            output.add_(bias.value().view({1, -1, 1, 1}));
        }

        return output;
    } else {
        // Use PyTorch's built-in implementation for larger batches/channels
        return at::conv_transpose2d(
            x, weight, bias,
            {stride, stride},
            {padding, padding},
            {output_padding, output_padding},
            groups
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Hybrid ConvTranspose2d forward (CUDA)");
}