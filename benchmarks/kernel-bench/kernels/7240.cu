#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(float* x, float* weight, float* bias, float* output, int stride, int padding, int dilation, int groups, int x_height, int x_width, int weight_height, int weight_width) {
    // Kernel implementation here
    // This is a placeholder for the actual convolution computation
}

void launch_conv2d_kernel(float* x, float* weight, float* bias, float* output, int stride, int padding, int dilation, int groups, int x_height, int x_width, int weight_height, int weight_width, cudaStream_t stream) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((x_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (x_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv2d_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(x, weight, bias, output, stride, padding, dilation, groups, x_height, x_width, weight_height, weight_width);
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    auto output = torch::empty({x.size(0), weight.size(0), (x.size(2) - weight.size(2) + 2 * padding) / stride + 1, (x.size(3) - weight.size(3) + 2 * padding) / stride + 1}, x.options());

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float* x_ptr = x.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    launch_conv2d_kernel(x_ptr, weight_ptr, bias_ptr, output_ptr, stride, padding, dilation, groups, x.size(2), x.size(3), weight.size(2), weight.size(3), stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with streams");
}