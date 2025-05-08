#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

// Utility macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses a balanced 3D grid to distribute workload evenly.
// gridDim.z covers the combined batch and output channels (b and oc), while gridDim.x and gridDim.y cover output width and height.
// Each thread computes one output element, ensuring each thread does roughly the same amount of work.
__global__ void conv2d_balanced_kernel(
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

    // Map blockIdx.z to batch index (b) and output channel (oc)
    int linear_idx = blockIdx.z;
    int b = linear_idx / out_channels;
    int oc = linear_idx % out_channels;

    // Compute output spatial coordinates using 2D grid and thread indices
    int ow = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    int oh = blockIdx.y * BLOCK_HEIGHT + threadIdx.y;

    if (ow >= out_width || oh >= out_height) return;

    float sum = 0.0f;

    // Determine group information for grouped convolution
    int group_out_channels = out_channels / groups;  // number of output channels per group
    int group = oc / group_out_channels;
    int in_channels_per_group = in_channels / groups;

    // Compute the convolution sum over the proper input channels and kernel window
    for (int c = 0; c < in_channels_per_group; ++c) {
        int ic = group * in_channels_per_group + c;
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int in_y = oh * stride - padding + kh * dilation;
                int in_x = ow * stride - padding + kw * dilation;
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = ((b * in_channels + ic) * in_height + in_y) * in_width + in_x;
                    int weight_idx = (((oc * in_channels_per_group + c) * kernel_height) + kh) * kernel_width + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
    output[output_idx] = sum;
}

// forward function sets up grid dimensions based on the output spatial size and the combined batch and channel dimension.
// This ensures that the workload is evenly partitioned across threads and blocks.

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

    // Calculate output dimensions using standard convolution formula
    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = x.options();
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    // Define block and grid dimensions for balanced workload distribution
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    dim3 grid(
        (out_width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
        (out_height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
        batch_size * out_channels  // each (b, oc) pair gets its own slice
    );

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv2d_balanced_kernel<<<grid, block>>>(
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
    m.def("forward", &forward, "CUDA 2D Convolution with Balanced Workload Distribution");
}
