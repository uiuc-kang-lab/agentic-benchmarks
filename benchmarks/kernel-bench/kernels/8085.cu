#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel for performing transposed 1D convolution with support for groups
__global__ void conv_transposed_1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int input_length,
    int output_length,
    int groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = output_length * out_channels;
    if (idx >= total_threads) return;

    // Determine spatial and channel indices
    int out_pos = idx / out_channels; // spatial position in output
    int out_ch = idx % out_channels;  // output channel

    // Handle groups
    int out_channels_per_group = out_channels / groups;
    int group = out_ch / out_channels_per_group;
    int in_channels_per_group = in_channels / groups;
    int out_ch_within_group = out_ch % out_channels_per_group;
    
    float sum = 0.0f;
    // Loop over the input channels within the correct group and over kernel positions
    for (int in_ch = 0; in_ch < in_channels_per_group; ++in_ch) {
        for (int k = 0; k < kernel_size; ++k) {
            int computed = out_pos + padding - k;
            // The stride condition: only accumulate when the index aligns
            if (computed % stride == 0) {
                int in_pos = computed / stride;
                if (in_pos >= 0 && in_pos < input_length) {
                    // Correct indexing for grouped convolution
                    int input_idx = (in_pos * in_channels) + (group * in_channels_per_group) + in_ch;
                    int weight_idx = (out_ch_within_group * in_channels_per_group * kernel_size) + 
                                   (in_ch * kernel_size) + k + 
                                   (group * out_channels_per_group * in_channels_per_group * kernel_size);
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    output[idx] = sum;
}

// Kernel to add bias to the output
__global__ void add_bias_kernel(float* output, const float* bias, int out_channels, int output_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = output_length * out_channels;
    if (idx < total) {
        int c = idx % out_channels;
        output[idx] += bias[c];
    }
}

// Forward function launching our custom kernel with dynamic block size selection
torch::Tensor forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride_,
    int64_t padding_,
    int64_t output_padding_,
    int64_t groups) {

    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    int stride = static_cast<int>(stride_);
    int padding = static_cast<int>(padding_);
    int output_padding = static_cast<int>(output_padding_);
    
    auto input_sizes = input.sizes();
    int N = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_length = input_sizes[2];
    
    // Assume weight shape is [out_channels, in_channels/groups, kernel_size]
    auto weight_sizes = weight.sizes();
    int out_channels = weight_sizes[0];
    int kernel_size = weight_sizes[2];
    
    // Calculate the output length for transposed convolution
    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({N, out_channels, output_length}, input.options());

    // Total number of work items per batch element
    int total_work = output_length * out_channels;
    
    // Experiment with block sizes based on total_work
    int BLOCK_SIZE = 256;  // default
    if (total_work <= 512) {
        BLOCK_SIZE = 32;
    } else if (total_work <= 1024) {
        BLOCK_SIZE = 64;
    } else if (total_work <= 4096) {
        BLOCK_SIZE = 128;
    } else if (total_work <= 16384) {
        BLOCK_SIZE = 256;
    } else {
        BLOCK_SIZE = 512;
    }
    
    int grid_x = (total_work + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_x, 1, N);
    dim3 block(BLOCK_SIZE);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch our convolution kernel for each batch element
    for (int n = 0; n < N; ++n) {
        float* output_ptr = output.data_ptr<float>() + n * out_channels * output_length;
        const float* input_ptr = input.data_ptr<float>() + n * in_channels * input_length;
        conv_transposed_1d_kernel<<<grid, block, 0, stream>>>(
            input_ptr,
            weight.data_ptr<float>(),
            output_ptr,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            input_length,
            output_length,
            static_cast<int>(groups)
        );
    }
    
    // If a bias tensor is provided, add it using a separate kernel
    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        int bias_total = output_length * out_channels;
        int bias_block = 256;
        int bias_grid = (bias_total + bias_block - 1) / bias_block;
        for (int n = 0; n < N; ++n) {
            float* output_ptr = output.data_ptr<float>() + n * out_channels * output_length;
            add_bias_kernel<<<bias_grid, bias_block, 0, stream>>>(
                output_ptr, 
                bias_tensor.data_ptr<float>(), 
                out_channels, 
                output_length
            );
        }
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Dynamic BlockSize Transposed 1D convolution forward (CUDA)");
}
