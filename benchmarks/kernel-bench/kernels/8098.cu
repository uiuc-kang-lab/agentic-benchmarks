#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__device__ __inline__ void compute_output_element(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int n, int c_out, int out_x,
    int in_channels, int out_channels,
    int in_length, int out_length,
    int kernel_size, int stride,
    int padding, int groups) {
    
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    const int group = c_out / out_channels_per_group;
    const int c_out_local = c_out % out_channels_per_group;

    scalar_t sum = 0;
    const int total_iters = in_channels_per_group * kernel_size;

    for (int idx = threadIdx.x % 32; idx < total_iters; idx += 32) {
        const int channel_local = idx / kernel_size;
        const int k = idx % kernel_size;
        const int in_channel = group * in_channels_per_group + channel_local;

        const int shifted = out_x + padding - k;
        if (shifted % stride == 0) {
            const int in_x = shifted / stride;
            if (in_x >= 0 && in_x < in_length) {
                const int input_idx = n * in_channels * in_length + in_channel * in_length + in_x;
                const int weight_idx = in_channel * out_channels_per_group * kernel_size 
                                     + c_out_local * kernel_size + k;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (threadIdx.x % 32 == 0) {
        output[n * out_channels * out_length + c_out * out_length + out_x] = sum;
    }
}

template <typename scalar_t>
__global__ void conv_transposed_1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_length, int out_length,
    int kernel_size, int stride,
    int padding, int groups) {
    
    const int elements_per_block = blockDim.x / 32;
    const int base_idx = blockIdx.x * elements_per_block;

    for (int i = 0; i < elements_per_block; ++i) {
        const int linear_idx = base_idx + i;
        if (linear_idx >= batch_size * out_channels * out_length) return;

        const int out_x = linear_idx % out_length;
        const int c_out = (linear_idx / out_length) % out_channels;
        const int n = linear_idx / (out_length * out_channels);

        compute_output_element(
            input, weight, output,
            n, c_out, out_x,
            in_channels, out_channels,
            in_length, out_length,
            kernel_size, stride,
            padding, groups
        );

        if (bias && threadIdx.x % 32 == 0) {
            output[linear_idx] += bias[c_out];
        }
    }
}

torch::Tensor forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) CHECK_INPUT(bias.value());

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_length = input.size(2);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1) * groups;
    const int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::empty({batch_size, out_channels, out_length}, input.options());
    
    const int threads_per_block = 256;
    const int elements_per_block = threads_per_block / 32;
    const int total_elements = batch_size * out_channels * out_length;
    const int blocks = (total_elements + elements_per_block - 1) / elements_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transposed_1d", ([&] {
        conv_transposed_1d_kernel<scalar_t><<<blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            in_length, out_length,
            kernel_size, stride,
            padding, groups
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Transposed 1D Conv with Warp-per-Output");
}