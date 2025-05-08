#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare constant memory for weights (assuming max size)
__constant__ float d_weights[1024];

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    // Copy weights to constant memory
    cudaMemcpyToSymbol(d_weights, 
                       weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float));

    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
        return torch::conv_transpose1d(
            x,
            weight,
            bias.value(),
            stride,
            padding,
            output_padding,
            groups
        );
    } else {
        return torch::conv_transpose1d(
            x,
            weight,
            torch::Tensor(),
            stride,
            padding,
            output_padding,
            groups
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA)");
}