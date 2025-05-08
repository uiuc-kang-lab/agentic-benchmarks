#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel flattens the output tensor so that each thread computes one output element.
// A grid-stride loop is used to evenly distribute the workload across threads and blocks.
// The kernel correctly handles grouped convolution with optional bias in full precision.
__global__ void conv2d_flattened_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr if not provided
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_global = blockDim.x * gridDim.x;

    while (idx < total) {
        // Decode the flat index into (b, oc, h, w) coordinates
        int w = idx % out_width;
        int tmp = idx / out_width;
        int h = tmp % out_height;
        tmp = tmp / out_height;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        float sum = 0.0f;
        int group_size = out_channels / groups;
        int group = oc / group_size;
        int in_channels_per_group = in_channels / groups;

        // Iterate over the input channels for this group and the kernel window
        for (int c = 0; c < in_channels_per_group; ++c) {
            int input_channel = group * in_channels_per_group + c;
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int in_y = h * stride - padding + kh * dilation;
                    int in_x = w * stride - padding + kw * dilation;
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                        int weight_idx = (((oc * in_channels_per_group + c) * kernel_height) + kh) * kernel_width + kw;
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

        idx += stride_global;
    }
}

// The forward function sets up the dimensions and launches the kernel using an even workload distribution
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

    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width  = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = x.options();
    torch::Tensor output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);
    float* output_ptr = output.data_ptr<float>();

    int total = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_flattened_kernel<<<blocks, threads>>>(
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

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA 2D Convolution with Even Workload Distribution");
}
