#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)
#define TILE_DIM 32

__global__ void depthwise_conv2d_warp_kernel(
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
    // Calculate indices
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int batch_channel = blockIdx.z;
    const int batch_idx = batch_channel / out_channels;
    const int out_ch = batch_channel % out_channels;
    
    // Calculate input channel and weight offset
    const int in_ch = out_ch / channels_per_group;
    const int weight_ch = out_ch % channels_per_group;

    // Calculate output coordinates
    const int out_x_base = blockIdx.x * TILE_DIM;
    const int out_y_base = blockIdx.y * TILE_DIM;
    
    // Shared memory for input tile and weights
    __shared__ float s_weight[WARPS_PER_BLOCK][9]; // Assuming max kernel_size = 3
    
    // Load weights into shared memory (only once per warp)
    if (lane_id < kernel_size * kernel_size) {
        const int weight_offset = in_ch * (channels_per_group * kernel_size * kernel_size) +
                                weight_ch * (kernel_size * kernel_size);
        s_weight[warp_id][lane_id] = weight[weight_offset + lane_id];
    }
    
    // Process TILE_DIM/WARPS_PER_BLOCK rows per warp
    const int rows_per_warp = TILE_DIM / WARPS_PER_BLOCK;
    const int out_y = out_y_base + warp_id * rows_per_warp + lane_id / (TILE_DIM/rows_per_warp);
    const int out_x = out_x_base + lane_id % (TILE_DIM/rows_per_warp);
    
    if (out_y < output_h && out_x < output_w) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            const int in_y = out_y * stride + ky - padding;
            
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int in_x = out_x * stride + kx - padding;
                
                if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
                    const int in_idx = batch_idx * (in_channels * input_h * input_w) +
                                     in_ch * (input_h * input_w) +
                                     in_y * input_w + in_x;
                    const float in_val = input[in_idx];
                    const float w_val = s_weight[warp_id][ky * kernel_size + kx];
                    sum += in_val * w_val;
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[out_ch];
        }
        
        const int out_idx = batch_idx * (out_channels * output_h * output_w) +
                           out_ch * (output_h * output_w) +
                           out_y * output_w + out_x;
        output[out_idx] = sum;
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
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int kernel_size = weight.size(2);
    const int channels_per_group = weight.size(1);
    const int out_channels = in_channels * channels_per_group;
    
    const int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());
    
    // Launch configuration
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (output_w + TILE_DIM - 1) / TILE_DIM,
        (output_h + TILE_DIM - 1) / TILE_DIM,
        batch_size * out_channels
    );
    
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    
    depthwise_conv2d_warp_kernel<<<grid, block>>>(
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