#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// This kernel performs a 2D transposed convolution operation with loop unrolling
// to reduce loop overhead. It computes the output for a single batch element and
// a single output channel. Grid dimension blockIdx.z encodes (batch, output_channel).
__global__ void conv_transpose2d_unroll_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int inChannels,
    int outChannels,
    int inH, int inW,
    int outH, int outW,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int padH, int padW
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_idx = blockIdx.z;
    int out_channel = global_idx % outChannels;
    int b = global_idx / outChannels;

    if (out_x >= outW || out_y >= outH) return;

    float sum = 0.0f;
    // Loop over input channels
    for (int in_channel = 0; in_channel < inChannels; in_channel++) {
        #pragma unroll
        for (int k_y = 0; k_y < kernelH; k_y++) {
            #pragma unroll
            for (int k_x = 0; k_x < kernelW; k_x++) {
                int in_y = out_y + padH - k_y;
                int in_x = out_x + padW - k_x;
                // Only consider valid contributions that align with stride
                if (in_y % strideH == 0 && in_x % strideW == 0) {
                    int r = in_y / strideH;
                    int c = in_x / strideW;
                    if (r >= 0 && r < inH && c >= 0 && c < inW) {
                        int input_idx = ((b * inChannels + in_channel) * inH + r) * inW + c;
                        int weight_idx = ((in_channel * outChannels + out_channel) * kernelH + k_y) * kernelW + k_x;
                        sum = __fmaf_rn(input[input_idx], weight[weight_idx], sum);
                    }
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[out_channel];
    }
    int output_idx = ((b * outChannels + out_channel) * outH + out_y) * outW + out_x;
    output[output_idx] = sum;
}

// Host function: wraps the kernel launch. The tensor dimensions are assumed as follows:
// input: [batch, inChannels, inH, inW]
// weight: [inChannels, outChannels, kernelH, kernelW]
// The output tensor is sized accordingly for a 2D transposed convolution.

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    pybind11::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    const float* bias_ptr = nullptr;
    torch::Tensor bias;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        bias_ptr = bias.data_ptr<float>();
    }

    int batch = input.size(0);
    int inChannels = input.size(1);
    int inH = input.size(2);
    int inW = input.size(3);
    
    int outChannels = weight.size(1);
    int kernelH = weight.size(2);
    int kernelW = weight.size(3);
    
    int strideH = stride[0];
    int strideW = stride[1];
    int padH = padding[0];
    int padW = padding[1];
    
    // Calculate output dimensions for the transposed convolution
    int outH = (inH - 1) * strideH - 2 * padH + kernelH;
    int outW = (inW - 1) * strideW - 2 * padW + kernelW;
    
    auto output = torch::zeros({batch, outChannels, outH, outW}, input.options());
    
    // Define block and grid dimensions. The grid z-dimension encodes (batch * outChannels).
    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x,
              (outH + block.y - 1) / block.y,
              batch * outChannels);
              
    conv_transpose2d_unroll_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
        inChannels,
        outChannels,
        inH, inW,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        padH, padW
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with loop unrolling",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
