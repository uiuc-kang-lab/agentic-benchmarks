#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define KERNEL_SIZE 3

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel using grid-stride loops to handle output workloads larger than the thread count
__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {

    int total = batch * out_channels * out_height * out_width;
    
    // Grid-stride loop over all output elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        // Decode the flattened index into (b, oc, oh, ow)
        int ow = idx % out_width;
        int tmp = idx / out_width;
        int oh = tmp % out_height;
        tmp = tmp / out_height;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        // For grouped convolution
        int group_in_channels = in_channels / groups;
        int out_channels_per_group = out_channels / groups;
        int group_idx = oc / out_channels_per_group;
        int input_channel_offset = group_idx * group_in_channels;

        float sum = (bias != nullptr) ? bias[oc] : 0.0f;

        // Loop over the relevant input channels and kernel window
        for (int ic = 0; ic < group_in_channels; ic++) {
            int input_channel = input_channel_offset + ic;
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                int h_in = oh * stride - padding + kh * dilation;
                if (h_in < 0 || h_in >= in_height) continue;
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    int w_in = ow * stride - padding + kw * dilation;
                    if (w_in < 0 || w_in >= in_width) continue;
                    int input_index = ((b * in_channels + input_channel) * in_height + h_in) * in_width + w_in;
                    int weight_index = ((oc * group_in_channels + ic) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw;
                    sum += input[input_index] * weight[weight_index];
                }
            }
        }
        output[idx] = sum;
    }
}

// Host function to setup and launch the CUDA kernel
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

    // Get tensor dimensions
    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    
    // Weight shape assumed: [out_channels, in_channels/groups, KERNEL_SIZE, KERNEL_SIZE]
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // Assumed square kernel
    TORCH_CHECK(kernel_size == KERNEL_SIZE, "Kernel size mismatch with defined KERNEL_SIZE");

    // Compute the output dimensions
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());
    
    int total = batch * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        stride,
        padding,
        dilation,
        groups
    );

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution using grid-stride loops");
}
