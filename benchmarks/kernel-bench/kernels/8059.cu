#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function for main convolution computation
__global__ void compute_conv_transpose(
    const float* __restrict__ input, 
    const float* __restrict__ weight,
    float* __restrict__ output, 
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    int output_padding, int input_length, int output_length) {
    
    const int n = blockIdx.z;
    const int out_ch = blockIdx.y;
    const int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_pos < output_length) {
        float sum = 0.0f;
        
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            for (int k = 0; k < kernel_size; ++k) {
                int in_pos = (out_pos + padding - kernel_size + 1 + k) / stride;
                if (in_pos >= 0 && in_pos < input_length && 
                    (out_pos + padding - k) % stride == 0) {
                    sum += input[n * in_channels * input_length + in_ch * input_length + in_pos] * 
                           weight[out_ch * in_channels * kernel_size + in_ch * kernel_size + k];
                }
            }
        }
        output[n * out_channels * output_length + out_ch * output_length + out_pos] = sum;
    }
}

// Host function interfaced via pybind11 that prepares inputs and launches the kernel
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

    // Extract dimensions from input and weight tensors
    auto input_sizes = input.sizes();  // [N, in_channels, input_width]
    int N = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_length = input_sizes[2];
    
    auto weight_sizes = weight.sizes(); // [in_channels, out_channels, kernel_size]
    int out_channels = weight_sizes[1];
    int kernel_size = weight_sizes[2];

    // Compute the output width as per conv_transpose1d formula
    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({N, out_channels, output_length}, input.options());
    
    // Setup launch configuration with a 3D grid: (output spatial dimension, output channels, batch)
    const int threads = 256;
    const int blocks_x = (output_length + threads - 1) / threads;
    dim3 blocks(blocks_x, out_channels, N);
    
    // Get the current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    compute_conv_transpose<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        input_length,
        output_length
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced workload Transposed 1D convolution forward (CUDA)");
}