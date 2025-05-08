#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv_transposed_1d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int in_channels, const int out_channels,
    const int input_width, const int output_width,
    const int kernel_size, const int stride,
    const int padding, const int output_padding,
    const int groups) {
    
    // Block handles multiple output elements for better efficiency
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.z;
    const int out_ch = blockIdx.y;
    const int out_x_base = blockIdx.x * blockDim.x;
    
    // Each thread processes multiple output positions
    const int out_x = out_x_base + tid;
    if (out_x >= output_width) return;
    const int out_x_pad = out_x + padding;
    
    // Calculate group information
    const int out_ch_per_group = out_channels / groups;
    const int in_ch_per_group = in_channels / groups;
    const int group = out_ch / out_ch_per_group;
    const int in_ch_start = group * in_ch_per_group;
    
    // Initialize output with bias if present
    float sum = (bias != nullptr) ? bias[out_ch] : 0.0f;
    
    // Pre-compute constant offsets
    const int batch_offset = batch_idx * in_channels * input_width;
    const int weight_ch_offset = out_ch * kernel_size;
    
    // Main convolution loop
    #pragma unroll 4
    for (int in_ch = 0; in_ch < in_ch_per_group; ++in_ch) {
        const int global_in_ch = in_ch_start + in_ch;
        const int input_ch_offset = batch_offset + global_in_ch * input_width;
        const int weight_offset = global_in_ch * out_channels * kernel_size + weight_ch_offset;
        
        #pragma unroll 2
        for (int k = 0; k < kernel_size; ++k) {
            const int in_x = (out_x + padding - k) / stride;
            if (in_x >= 0 && in_x < input_width && (out_x + padding - k) % stride == 0) {
                sum += input[input_ch_offset + in_x] * weight[weight_offset + k];
            }
        }
    }
    
    // Write output
    const int out_idx = batch_idx * out_channels * output_width + 
                       out_ch * output_width + out_x;
    output[out_idx] = sum;
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
    
    const int output_width = (input_width - 1) * stride - 2 * padding + 
                            kernel_size + output_padding;
    
    auto output = torch::zeros({N, out_channels, output_width}, input.options());
    
    const int BLOCK_SIZE = (output_width < 256) ? output_width : 256;
    const int grid_x = (output_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 grid(grid_x, out_channels, N);
    dim3 block(BLOCK_SIZE);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    conv_transposed_1d_optimized_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, in_channels, out_channels,
        input_width, output_width,
        kernel_size, stride,
        padding, output_padding,
        groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Transposed 1D convolution forward (CUDA)");
}