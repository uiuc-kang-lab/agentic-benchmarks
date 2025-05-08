#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void conv2d_balanced_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
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

    const int total_outputs = batch_size * out_channels * out_height * out_width;
    const int elements_per_thread = (total_outputs + (blockDim.x * gridDim.x) - 1) / (blockDim.x * gridDim.x);
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int group_out_channels = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;

    for (int i = 0; i < elements_per_thread; i++) {
        const int output_idx = thread_id + i * (blockDim.x * gridDim.x);
        if (output_idx >= total_outputs) break;

        const int b = output_idx / (out_channels * out_height * out_width);
        const int tmp1 = output_idx % (out_channels * out_height * out_width);
        const int oc = tmp1 / (out_height * out_width);
        const int tmp2 = tmp1 % (out_height * out_width);
        const int h = tmp2 / out_width;
        const int w = tmp2 % out_width;

        const int group = oc / group_out_channels;
        float sum = 0.0f;

        #pragma unroll 4
        for (int c = 0; c < in_channels_per_group; ++c) {
            const int input_channel = group * in_channels_per_group + c;
            
            #pragma unroll
            for (int kh = 0; kh < kernel_height; ++kh) {
                const int in_y = h * stride - padding + kh * dilation;
                if (in_y >= 0 && in_y < in_height) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        const int in_x = w * stride - padding + kw * dilation;
                        if (in_x >= 0 && in_x < in_width) {
                            const int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                            const int weight_idx = ((oc * in_channels_per_group + c) * kernel_height + kh) * kernel_width + kw;
                            sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += __ldg(&bias[oc]);
        }

        output[output_idx] = sum;
    }
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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = x.options();
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int max_blocks = 65535;
    const int grid_size = min(num_blocks, max_blocks);

    conv2d_balanced_kernel<<<grid_size, BLOCK_SIZE>>>(
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