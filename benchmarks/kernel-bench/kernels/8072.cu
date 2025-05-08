#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel for transposed 1D convolution combining best practices
__global__ void conv_transposed_1d_optimized_kernel(
    const float* __restrict__ input,   // [N, in_channels, input_width]
    const float* __restrict__ weight,  // [in_channels, out_channels, kernel_size]
    const float* __restrict__ bias,    // [out_channels] or nullptr
    float* __restrict__ output,        // [N, out_channels, output_width]
    const int N,
    const int in_channels,
    const int out_channels,
    const int input_width,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups) {

    // Each block processes a tile in the spatial (output) dimension
    const int out_x_base = blockIdx.x * blockDim.x;
    const int tid = threadIdx.x;
    const int out_x = out_x_base + tid;
    if (out_x >= output_width) return;

    // Map grid dimensions: blockIdx.y -> output channel, blockIdx.z -> batch
    const int oc = blockIdx.y;
    const int n  = blockIdx.z;

    // Determine group parameters
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    const int group = oc / out_channels_per_group;
    const int ic_start = group * in_channels_per_group;

    // Initialize accumulation; add bias if provided
    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Offset for the current batch
    const int batch_offset = n * in_channels * input_width;

    // Loop through each input channel in the group
    #pragma unroll
    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        int global_ic = ic_start + ic;
        const int input_channel_offset = batch_offset + global_ic * input_width;
        // Compute weight base for current input channel and output channel
        const int weight_base = global_ic * (out_channels * kernel_size) + oc * kernel_size;

        // Loop over kernel width
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int temp = out_x + padding - k;
            if (temp < 0) continue;
            if ((temp % stride) != 0) continue;
            int ix = temp / stride;
            if (ix >= input_width) continue;
            
            sum += input[input_channel_offset + ix] * weight[weight_base + k];
        }
    }

    // Write the result to the output tensor
    const int out_index = n * (out_channels * output_width) + oc * output_width + out_x;
    output[out_index] = sum;
}

// Host function: prepares the launch configuration and calls the CUDA kernel
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

    // Extract dimensions from input and weight tensors
    auto input_sizes = input.sizes();  // [N, in_channels, input_width]
    const int N = input_sizes[0];
    const int in_channels = input_sizes[1];
    const int input_width = input_sizes[2];

    auto weight_sizes = weight.sizes();  // [in_channels, out_channels, kernel_size]
    const int out_channels = weight_sizes[1];
    const int kernel_size = weight_sizes[2];

    // Calculate output width using the transposed convolution formula
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({N, out_channels, output_width}, input.options());

    // Configure kernel launch parameters
    const int BLOCK_SIZE = 128;  // Chosen for high throughput (e.g., on H100)
    const int blocks_x = (output_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(blocks_x, out_channels, N);
    dim3 block(BLOCK_SIZE);

    // Launch kernel on current CUDA stream
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
    m.def("forward", &forward, "Optimized Transposed 1D Convolution forward (CUDA)");
}
