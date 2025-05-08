#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__global__ void convTranspose2DKernel(const float* input, const float* weight, float* output, int inputHeight, int inputWidth, int outputHeight, int outputWidth, int kernelHeight, int kernelWidth, int padHeight, int padWidth, int strideHeight, int strideWidth) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < outputWidth && out_y < outputHeight) {
        float value = 0.0;
        for (int ky = 0; ky < kernelHeight; ++ky) {
            for (int kx = 0; kx < kernelWidth; ++kx) {
                int in_y = out_y * strideHeight - padHeight + ky;
                int in_x = out_x * strideWidth - padWidth + kx;
                if (in_y >= 0 && in_y < inputHeight && in_x >= 0 && in_x < inputWidth) {
                    value += weight[ky * kernelWidth + kx] * input[in_y * inputWidth + in_x];
                }
            }
        }
        output[out_y * outputWidth + out_x] = value;
    }
}

void launchConvTranspose2D(const torch::Tensor& input, const torch::Tensor& weight, torch::Tensor& output, std::vector<int64_t> stride, std::vector<int64_t> padding) {
    const int block_size = 128;
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks((output.size(2) + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output.size(1) + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int inputHeight = input.size(1);
    int inputWidth = input.size(2);
    int outputHeight = output.size(1);
    int outputWidth = output.size(2);
    int kernelHeight = weight.size(2);
    int kernelWidth = weight.size(3);

    convTranspose2DKernel<<<numBlocks, threadsPerBlock>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        inputHeight, inputWidth, outputHeight, outputWidth, kernelHeight, kernelWidth,
        padding[0], padding[1], stride[0], stride[1]
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launchConvTranspose2D, "Optimized Conv Transpose 2D forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("stride"),
          py::arg("padding"));
}