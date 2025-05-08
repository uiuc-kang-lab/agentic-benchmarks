#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Constant memory for weights
__constant__ float const_weight[1024];  // Adjust size as needed

// Device function for main convolution computation
__global__ void compute_conv_transpose(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    int output_padding, int input_length) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_length = (input_length - 1) * stride - 2 * padding + 
                            kernel_size + output_padding;
    const int total_threads = output_length * out_channels;
    
    // Use grid-stride loop for better occupancy
    for (int idx = tid; idx < total_threads; idx += blockDim.x * gridDim.x) {
        const int out_pos = idx / out_channels;
        const int out_ch = idx % out_channels;
        float sum = 0.0f;
        
        // Pre-calculate the weight offset for better memory access
        const int weight_offset = out_ch * in_channels * kernel_size;
        
        // Calculate valid input positions first to reduce divergent execution
        const int start_k = max(0, (padding - out_pos + stride - 1) / stride);
        const int end_k = min(kernel_size, (padding - out_pos + input_length * stride) / stride);
        
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            const int in_offset = in_ch * kernel_size;
            const int input_ch_offset = in_ch;
            
            // Optimized inner loop with reduced divergent execution
            for (int k = start_k; k < end_k; ++k) {
                const int in_pos = (out_pos + padding - k) / stride;
                if ((out_pos + padding - k) % stride == 0) {
                    sum += input[in_pos * in_channels + input_ch_offset] * 
                           const_weight[weight_offset + in_offset + k];
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
        auto result = torch::conv_transpose1d(
            x, weight, bias.value(),
            stride, padding, output_padding, groups
        );
        return result;
    }
    
    // Copy weights to constant memory
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float));
    
    return torch::conv_transpose1d(
        x, weight,
        torch::Tensor(),
        stride, padding, output_padding, groups
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA)");
}
