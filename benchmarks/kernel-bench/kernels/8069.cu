#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv_transposed_1d_aligned_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,
    const int in_channels,
    const int out_channels,
    const int input_width,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups
) {
    const int tid = threadIdx.x;
    const int ox = blockIdx.x * blockDim.x + tid;
    const int pos = ox + padding;
    const int oc = blockIdx.y;
    const int n = blockIdx.z;

    if (ox >= output_width) return;

    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    const int group = oc / out_channels_per_group;
    const int ic_start = group * in_channels_per_group;

    float sum = (bias != nullptr) ? __ldg(&bias[oc]) : 0.0f;

    #pragma unroll 4
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        const int global_ic = ic_start + ic;
        const float* weight_ptr = weight + (global_ic * out_channels + oc) * kernel_size;
        
        #pragma unroll
        for (int k = 0; k < kernel_size; k++) {
            const int temp = ox + padding - k;
            if (temp >= 0 && (temp % stride) == 0) {
                const int ix = temp / stride;
                if (ix < input_width) {
                    const float in_val = __ldg(&input[n * (in_channels * input_width) + 
                                                     global_ic * input_width + ix]);
                    const float w_val = __ldg(&weight_ptr[k]);
                    sum += in_val * w_val;
                }
            }
        }
    }

    output[n * (out_channels * output_width) + oc * output_width + ox] = sum;
}

torch::Tensor forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    const auto input_sizes = input.sizes();
    const int N = input_sizes[0];
    const int in_channels = input_sizes[1];
    const int input_width = input_sizes[2];

    const auto weight_sizes = weight.sizes();
    const int out_channels = weight_sizes[1];
    const int kernel_size = weight_sizes[2];

    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({N, out_channels, output_width}, input.options());

    const int threads = 256;
    const int blocks_x = (output_width + threads - 1) / threads;
    dim3 blocks(blocks_x, out_channels, N);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    conv_transposed_1d_aligned_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, in_channels, out_channels,
        input_width, output_width,
        kernel_size, stride, padding,
        output_padding, groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Aligned Transposed 1D convolution forward (CUDA)");
}