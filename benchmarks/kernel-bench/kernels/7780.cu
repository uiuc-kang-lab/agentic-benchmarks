#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constants for tiling and thread block dimensions
#define TILE_WIDTH 16
#define TILE_HEIGHT 16 
#define BLOCK_SIZE 16
#define MAX_THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Shared memory tile declarations
__shared__ float input_tile[TILE_HEIGHT][TILE_WIDTH];
__shared__ float weight_tile[TILE_HEIGHT][TILE_WIDTH];

__global__ void conv2d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width, 
    int out_channels,
    int kernel_height,
    int kernel_width,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Calculate indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ow = blockIdx.x * blockDim.x + tx;
    int oh = blockIdx.y * blockDim.y + ty;
    
    // Map blockIdx.z to batch and output channel 
    int z_idx = blockIdx.z;
    int b = z_idx / out_channels;
    int oc = z_idx % out_channels;

    if (ow >= out_width || oh >= out_height) return;

    // Calculate group information
    int group_out_channels = out_channels / groups;
    int group = oc / group_out_channels;
    int in_channels_per_group = in_channels / groups;
    
    float sum = 0.0f;

    // Loop over input channels in tiles
    for (int ic_block = 0; ic_block < in_channels_per_group; ic_block += TILE_WIDTH) {
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
                
                // Load input tile collaboratively
                int in_y = oh * stride - padding + kh * dilation;
                int in_x = ow * stride - padding + kw * dilation;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    for (int t = 0; t < TILE_WIDTH; t += WARP_SIZE) {
                        int ic = ic_block + (tx + t) % TILE_WIDTH;
                        if (ic < in_channels_per_group) {
                            int input_idx = ((b * in_channels + group * in_channels_per_group + ic) * 
                                           in_height + in_y) * in_width + in_x;
                            input_tile[ty][tx] = input[input_idx];
                        }
                    }
                }
                
                // Load weight tile collaboratively  
                for (int t = 0; t < TILE_WIDTH; t += WARP_SIZE) {
                    int ic = ic_block + (tx + t) % TILE_WIDTH;
                    if (ic < in_channels_per_group) {
                        int weight_idx = ((oc * in_channels_per_group + ic) * 
                                        kernel_height + kh) * kernel_width + kw;
                        weight_tile[ty][tx] = weight[weight_idx];
                    }
                }
                
                __syncthreads();
                
                // Compute partial sum for the tile
                #pragma unroll
                for (int i = 0; i < TILE_WIDTH && (ic_block + i) < in_channels_per_group; i++) {
                    sum += input_tile[ty][i] * weight_tile[ty][i];
                }
                
                __syncthreads();
            }
        }
    }

    // Add bias and write output
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    if (ow < out_width && oh < out_height) {
        int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
        output[output_idx] = sum;
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

    auto dims = x.sizes();
    int batch_size = dims[0];
    int in_channels = dims[1];
    int in_height = dims[2];
    int in_width = dims[3];
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size * out_channels
    );

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    conv2d_optimized_kernel<<<grid, block>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_height, kernel_width,
        out_height, out_width, stride, padding, dilation, groups
    );

    return output;
}