#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel function: optimized 2D convolution using shared memory and warp reduction
__global__ void conv2d_optimized_kernel(
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

    extern __shared__ float shared_input[];
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_height * out_width;

    if (output_idx < total) {
        int w = output_idx % out_width;
        int h = (output_idx / out_width) % out_height;
        int oc = (output_idx / (out_width * out_height)) % out_channels;
        int b = output_idx / (out_channels * out_height * out_width);

        float sum = 0.0f;
        int group_out_channels = out_channels / groups;
        int group = oc / group_out_channels;
        int in_channels_per_group = in_channels / groups;

        // Load to shared memory to avoid repeated global memory access
        for (int kc = threadIdx.x; kc < in_channels_per_group * kernel_height * kernel_width; kc += blockDim.x) {
            int c = kc / (kernel_height * kernel_width);
            int kh_kw = kc % (kernel_height * kernel_width);
            int kh = kh_kw / kernel_width;
            int kw = kh_kw % kernel_width;
            int in_x = w * stride - padding + kw * dilation;
            int in_y = h * stride - padding + kh * dilation;
            int input_chan = group * in_channels_per_group + c;
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                shared_input[kc] = input[(((b * in_channels + input_chan) * in_height + in_y) * in_width + in_x)];
            } else {
                shared_input[kc] = 0.0f;
            }
        }
        __syncthreads();

        // Now perform convolution with shared memory
        for (int c = 0; c < in_channels_per_group; ++c) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int weight_idx = ((oc * in_channels_per_group + c) * kernel_height + kh) * kernel_width + kw;
                    sum += shared_input[(c * kernel_height + kh) * kernel_width + kw] * weight[weight_idx];
                }
            }
        }

        // Warp-level reduction
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (threadIdx.x % warpSize == 0) {
            if (bias != nullptr) {
                sum += bias[oc];
            }
            output[output_idx] = sum;
        }
    }
}

// Forward function to execute optimized kernel
// Uses shared memory and warp-level primitive operations

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
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    int total = batch_size * out_channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    size_t shared_mem_size = in_channels * kernel_height * kernel_width * sizeof(float) / groups;

    conv2d_optimized_kernel<<<grid_size, block_size, shared_mem_size>>>(
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
    m.def("forward", &forward, "CUDA 2D Convolution with Shared Memory and Warp-Level Reduction");
}