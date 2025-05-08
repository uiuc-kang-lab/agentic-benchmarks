#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_uniform_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation,
    const int groups) {

    // Calculate output position
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z % out_channels;
    const int b = blockIdx.z / out_channels;

    // Early exit if outside output bounds
    if (w >= out_width || h >= out_height || b >= batch_size) {
        return;
    }

    // Pre-compute group information
    const int group_out_channels = out_channels / groups;
    const int group = oc / group_out_channels;
    const int in_channels_per_group = in_channels / groups;
    const int group_in_offset = group * in_channels_per_group;

    // Pre-compute input positions
    const int in_y_start = h * stride - padding;
    const int in_x_start = w * stride - padding;

    // Calculate valid kernel bounds
    const int kh_start = max(0, (-in_y_start + dilation - 1) / dilation);
    const int kw_start = max(0, (-in_x_start + dilation - 1) / dilation);
    const int kh_end = min(kernel_height, (in_height - in_y_start + dilation - 1) / dilation);
    const int kw_end = min(kernel_width, (in_width - in_x_start + dilation - 1) / dilation);

    float sum = 0.0f;

    // Main computation loop with minimal branching
    #pragma unroll 4
    for (int c = 0; c < in_channels_per_group; ++c) {
        const int input_channel = group_in_offset + c;
        const float* input_ptr = input + ((b * in_channels + input_channel) * in_height * in_width);
        const float* weight_ptr = weight + ((oc * in_channels_per_group + c) * kernel_height * kernel_width);

        for (int kh = kh_start; kh < kh_end; ++kh) {
            const int in_y = in_y_start + kh * dilation;
            
            for (int kw = kw_start; kw < kw_end; ++kw) {
                const int in_x = in_x_start + kw * dilation;
                sum += input_ptr[in_y * in_width + in_x] * weight_ptr[kh * kernel_width + kw];
            }
        }
    }

    // Add bias if present (uniform operation across warp)
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // Write output (coalesced write)
    output[((b * out_channels + oc) * out_height + h) * out_width + w] = sum;
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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    // Choose block size to maximize occupancy while maintaining coalesced access
    const dim3 block_size(32, 8);
    const dim3 grid_size(
        (out_width + block_size.x - 1) / block_size.x,
        (out_height + block_size.y - 1) / block_size.y,
        batch_size * out_channels
    );

    conv2d_uniform_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
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
    m.def("forward", &forward, "CUDA 2D Convolution with Uniform Control Flow");
}