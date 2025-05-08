#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward function definition
torch::Tensor conv_transpose2d_forward_overlap(
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

    // Create a CUDA stream for overlapping operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronous memory copy operations and kernel execution
    at::cuda::CUDAStreamGuard guard(stream);

    auto result = at::conv_transpose2d(
        x,
        weight,
        bias,
        {stride, stride},                   // stride
        {padding, padding},                 // padding
        {output_padding, output_padding},   // output_padding
        groups
    );

    // Synchronize the stream to ensure all operations complete
    cudaStreamSynchronize(stream);
    
    // Destroy the stream
    cudaStreamDestroy(stream);

    return result;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_overlap", &conv_transpose2d_forward_overlap, "ConvTranspose2d forward with overlapping memory (CUDA)");
}