#include <torch/extension.h>
#include <vector>

__global__ void conv_transpose2d_unroll_kernel(float* x, float* weight, float* output,
                                                int width, int height, int kernel_width, int kernel_height,
                                                int stride_x, int stride_y, int padding_x, int padding_y) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    float result = 0.0f;

    #pragma unroll
    for (int ky = 0; ky < kernel_height; ++ky) {
        int in_y = out_y + ky - padding_y;
        if (in_y >= 0 && in_y < height) {
            #pragma unroll
            for (int kx = 0; kx < kernel_width; ++kx) {
                int in_x = out_x + kx - padding_x;
                if (in_x >= 0 && in_x < width) {
                    result += x[in_y * width + in_x] * weight[ky * kernel_width + kx];
                }
            }
        }
    }
    if (out_x < width && out_y < height) { output[out_y * width + out_x] = result; }
}

torch::Tensor conv_transpose2d_cuda_unroll(
    torch::Tensor x,
    torch::Tensor weight,
    std::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    auto output = torch::zeros_like(x);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((x.size(1) + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (x.size(0) + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conv_transpose2d_unroll_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        x.size(1),
        x.size(0),
        weight.size(1),
        weight.size(0),
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda_unroll, "ConvTranspose2D forward with unrolled loops (CUDA)");
}