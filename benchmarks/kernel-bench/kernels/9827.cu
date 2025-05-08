#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define PIXELS_PER_THREAD 4

__global__ void depthwise_conv2d_coalesced_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int input_h,
    const int input_w,
    const int out_channels,
    const int output_h,
    const int output_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int channels_per_group
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_pixels = output_h * output_w;
    const int pixels_per_block = BLOCK_SIZE * PIXELS_PER_THREAD;
    const int total_pixels = batch_size * out_channels * num_pixels;
    
    for (int pixel_offset = bid * pixels_per_block + tid; 
         pixel_offset < total_pixels; 
         pixel_offset += pixels_per_block * gridDim.x) {
        
        const int pixel_idx = pixel_offset % num_pixels;
        const int temp = pixel_offset / num_pixels;
        const int oc = temp % out_channels;
        const int b = temp / out_channels;
        
        const int out_x = pixel_idx % output_w;
        const int out_y = pixel_idx / output_w;
        
        const int in_ch = oc / channels_per_group;
        const int weight_ch = oc % channels_per_group;
        
        const int input_batch_offset = b * (in_channels * input_h * input_w);
        const int input_channel_offset = in_ch * (input_h * input_w);
        
        const int weight_offset = in_ch * (channels_per_group * kernel_size * kernel_size) +
                                weight_ch * (kernel_size * kernel_size);
        
        float sum = 0.0f;
        
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            const int in_y = out_y * stride + ky - padding;
            if (in_y >= 0 && in_y < input_h) {
                const int in_y_offset = in_y * input_w;
                const int weight_y_offset = ky * kernel_size;
                
                #pragma unroll
                for (int kx = 0; kx < kernel_size; ++kx) {
                    const int in_x = out_x * stride + kx - padding;
                    if (in_x >= 0 && in_x < input_w) {
                        const float in_val = input[input_batch_offset + 
                                                 input_channel_offset + 
                                                 in_y_offset + in_x];
                        const float w_val = weight[weight_offset + 
                                                 weight_y_offset + kx];
                        sum += in_val * w_val;
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        output[pixel_offset] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int kernel_size = weight.size(2);
    const int channels_per_group = weight.size(1);
    const int out_channels = in_channels * channels_per_group;
    
    const int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, 
                             input.options());
    
    const int total_pixels = batch_size * out_channels * output_h * output_w;
    const int pixels_per_block = BLOCK_SIZE * PIXELS_PER_THREAD;
    const int num_blocks = (total_pixels + pixels_per_block - 1) / pixels_per_block;
    
    const int max_blocks = 65535;
    const int grid_dim = std::min(num_blocks, max_blocks);
    
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    
    depthwise_conv2d_coalesced_kernel<<<grid_dim, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        output_h,
        output_w,
        kernel_size,
        stride,
        padding,
        channels_per_group
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), 
          py::arg("stride"), py::arg("padding"));
}