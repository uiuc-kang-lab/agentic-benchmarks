#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cuda_runtime.h>

namespace py = pybind11;

// This kernel implements a basic 2D transposed convolution for one batch sample.
// It computes the output for a given batch sample and output channel, using standard indexing.
// The kernel assumes input tensor shape [inChannels, inH, inW] and weight shape [inChannels, outChannels, kernelH, kernelW].
// The bias (if provided) is applied per output channel.
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int inH, int inW,
    int outH, int outW,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int padH, int padW,
    int inChannels, int outChannels
) {
    // Calculate output x and y indices
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    // blockIdx.z is used to index output channel
    int out_channel = blockIdx.z;
    if (out_x < outW && out_y < outH && out_channel < outChannels) {
        float sum = 0.0f;
        // Loop over all input channels
        for (int in_channel = 0; in_channel < inChannels; ++in_channel) {
            // Loop over kernel height and width
            for (int k_y = 0; k_y < kernelH; ++k_y) {
                for (int k_x = 0; k_x < kernelW; ++k_x) {
                    // Compute corresponding input position
                    int in_y = out_y + padH - k_y;
                    int in_x = out_x + padW - k_x;
                    // For transposed convolution, the input contributes only if the position aligns with stride
                    if (in_y % strideH == 0 && in_x % strideW == 0) {
                        int r = in_y / strideH;
                        int c = in_x / strideW;
                        if (r >= 0 && r < inH && c >= 0 && c < inW) {
                            int input_idx = in_channel * (inH * inW) + r * inW + c;
                            int weight_idx = in_channel * (outChannels * kernelH * kernelW) + 
                                             out_channel * (kernelH * kernelW) + k_y * kernelW + k_x;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[out_channel];
        }
        int output_idx = out_channel * (outH * outW) + out_y * outW + out_x;
        output[output_idx] = sum;
    }
}


// The forward function partitions the batch dimension and launches the kernel on multiple CUDA streams
// to overlap kernel execution with any potential memory transfer overhead and maximize throughput.
// It assumes input tensor shape: [batch, inChannels, inH, inW]
// and weight tensor shape: [inChannels, outChannels, kernelH, kernelW].

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    c10::optional<torch::Tensor> bias = c10::nullopt;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
    }

    // Extract dimensions
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

    // Calculate output dimensions for transposed convolution
    int outH = (inH - 1) * strideH - 2 * padH + kernelH;
    int outW = (inW - 1) * strideW - 2 * padW + kernelW;

    auto output = torch::zeros({batch, outChannels, outH, outW}, input.options());

    // Create multiple CUDA streams to pipeline kernel executions for different batch samples
    int numStreams = 2; // Using two streams for overlapping
    std::vector<cudaStream_t> streams(numStreams);
    for (int s = 0; s < numStreams; ++s) {
        cudaStreamCreate(&streams[s]);
    }

    // Define kernel launch configuration
    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y, outChannels);

    // Launch kernels for each sample in the batch using round-robin assignment of streams
    for (int b = 0; b < batch; b++) {
        const float* input_ptr = input[b].data_ptr<float>();   // shape: [inChannels, inH, inW]
        float* output_ptr = output[b].data_ptr<float>();         // shape: [outChannels, outH, outW]
        const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
        cudaStream_t stream = streams[b % numStreams];

        conv_transpose2d_kernel<<<grid, block, 0, stream>>>(
            input_ptr,
            weight.data_ptr<float>(),
            bias_ptr,
            output_ptr,
            inH, inW,
            outH, outW,
            kernelH, kernelW,
            strideH, strideW,
            padH, padW,
            inChannels, outChannels
        );
    }

    // Synchronize and destroy streams
    for (int s = 0; s < numStreams; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with stream pipelining",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}
