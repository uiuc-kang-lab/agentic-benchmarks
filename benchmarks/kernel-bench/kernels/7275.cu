#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define KERNEL_SIZE 3
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {
    
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    
    // Calculate output position
    const int total_output_size = batch_size * out_channels * out_height * out_width;
    const int warps_needed = (total_output_size + WARP_SIZE - 1) / WARP_SIZE;
    
    if (global_warp_id < warps_needed) {
        const int output_idx = global_warp_id * WARP_SIZE + lane_id;
        
        if (output_idx < total_output_size) {
            // Decode output index
            const int w = output_idx % out_width;
            int temp = output_idx / out_width;
            const int h = temp % out_height;
            temp /= out_height;
            const int c = temp % out_channels;
            const int b = temp / out_channels;
            
            float sum = bias ? bias[c] : 0.0f;
            
            // Compute input window bounds
            const int h_start = h * stride - padding;
            const int w_start = w * stride - padding;
            
            // Process input channels
            for (int ic = 0; ic < in_channels; ic++) {
                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    const int h_in = h_start + kh;
                    
                    if (h_in >= 0 && h_in < in_height) {
                        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                            const int w_in = w_start + kw;
                            
                            if (w_in >= 0 && w_in < in_width) {
                                const float input_val = input[((b * in_channels + ic) * in_height + h_in) * in_width + w_in];
                                const float weight_val = weight[((c * in_channels + ic) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw];
                                sum += input_val * weight_val;
                            }
                        }
                    }
                }
            }
            
            // Write output
            output[output_idx] = sum;
        }
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
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto in_height = x.size(2);
    auto in_width = x.size(3);
    auto out_channels = weight.size(0);
    auto out_height = (in_height + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;
    auto out_width = (in_width + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int threads_per_block = 256; // 8 warps per block
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    conv2d_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        stride,
        padding
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution");
}