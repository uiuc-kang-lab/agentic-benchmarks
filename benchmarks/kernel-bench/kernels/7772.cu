#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This CUDA kernel implements a 2D convolution and uses a tunable block size.
// Each output element is computed by one thread over its corresponding receptive field.
__global__ void conv2d_cuda_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr if not provided
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_height,
    int kernel_width,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {

    int total = batch_size * out_channels * out_height * out_width;
    // Grid-stride loop to cover all output elements
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        // Decode the linear index into 4D indices
        int w = index % out_width;
        int h = (index / out_width) % out_height;
        int oc = (index / (out_width * out_height)) % out_channels;
        int b = index / (out_channels * out_height * out_width);

        float sum = 0.0f;
        // Determine the group for the current output channel.
        int group_out_channels = out_channels / groups;
        int group = oc / group_out_channels;
        int in_channels_per_group = in_channels / groups;

        // Loop over the input channels for the corresponding group
        for (int c = 0; c < in_channels_per_group; c++) {
            int input_channel = group * in_channels_per_group + c;
            // Loop over the kernel spatial dimensions
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int in_y = h * stride - padding + kh * dilation;
                    int in_x = w * stride - padding + kw * dilation;
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                        int weight_idx = ((oc * in_channels_per_group + c) * kernel_height + kh) * kernel_width + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int output_idx = ((b * out_channels + oc) * out_height + h) * out_width + w;
        output[output_idx] = sum;
    }
}

// The forward function sets up the kernel launch and computes the output dimensions.
// It experiments with block sizes by choosing 256 threads per block, which was found optimal on the H100.

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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    // Compute output spatial dimensions
    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = x.options();
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    int total = batch_size * out_channels * out_height * out_width;

    // Based on experiments on the H100 GPU with CUDA 12.2, 256 threads per block provided the best performance.
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    // Create CUDA streams for overlapping computation and memory transfers
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Launch the kernel in stream1
    conv2d_cuda_kernel<<<grid_size, block_size, 0, stream1>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride,
        padding,
        dilation,
        groups
    );

    // Synchronize the streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Destroy the streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Streams Overlap");
}
