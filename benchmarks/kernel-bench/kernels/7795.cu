#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Shared memory tile sizes
#define TILE_WIDTH 32
#define TILE_HEIGHT 16

template<bool HasBias>
__global__ void adaptive_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation,
    const int groups) {
    
    __shared__ float input_tile[TILE_HEIGHT][TILE_WIDTH];
    __shared__ float weight_tile[TILE_HEIGHT][TILE_WIDTH];
    
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z;
    
    if (w >= out_width || h >= out_height || oc >= out_channels) return;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // For grouped convolution
    const int group_out_channels = out_channels / groups;
    const int group = oc / group_out_channels;
    const int in_channels_per_group = in_channels / groups;
    
    float sum = 0.0f;
    
    // Loop over batches
    for (int b = 0; b < batch_size; ++b) {
        // Loop over input channels in the group
        for (int ic = 0; ic < in_channels_per_group; ic += TILE_WIDTH) {
            // Load input tile cooperatively
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    const int in_y = h * stride - padding + kh * dilation;
                    const int in_x = w * stride - padding + kw * dilation;
                    
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        const int input_channel = group * in_channels_per_group + ic + tx;
                        if (input_channel < in_channels) {
                            input_tile[ty][tx] = input[
                                ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x
                            ];
                        }
                    } else {
                        input_tile[ty][tx] = 0.0f;
                    }
                    
                    // Load weight tile
                    const int weight_channel = ic + tx;
                    if (weight_channel < in_channels_per_group) {
                        weight_tile[ty][tx] = weight[
                            (((oc * in_channels_per_group + weight_channel) * kernel_height) + kh) * kernel_width + kw
                        ];
                    } else {
                        weight_tile[ty][tx] = 0.0f;
                    }
                    
                    __syncthreads();
                    
                    // Compute partial sum using tiled data
                    #pragma unroll
                    for (int i = 0; i < TILE_WIDTH; ++i) {
                        sum += input_tile[ty][i] * weight_tile[ty][i];
                    }
                    
                    __syncthreads();
                }
            }
        }
        
        // Add bias if needed
        if (HasBias) {
            sum += bias[oc];
        }
        
        // Write output
        const int output_idx = ((b * out_channels + oc) * out_height + h) * out_width + w;
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
    
    // Get dimensions
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);
    
    // Compute output dimensions
    const int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
    
    // Create output tensor
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());
    
    // Get raw pointers
    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    // Set grid and block dimensions
    dim3 block_dim(TILE_WIDTH, TILE_HEIGHT, 1);
    dim3 grid_dim(
        (out_width + block_dim.x - 1) / block_dim.x,
        (out_height + block_dim.y - 1) / block_dim.y,
        out_channels
    );
    
    // Launch appropriate kernel based on bias presence
    if (bias.has_value()) {
        adaptive_conv2d_kernel<true><<<grid_dim, block_dim>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, in_height, in_width,
            out_channels, kernel_height, kernel_width,
            out_height, out_width, stride, padding, dilation, groups
        );
    } else {
        adaptive_conv2d_kernel<false><<<grid_dim, block_dim>>>(
            input_ptr, weight_ptr, nullptr, output_ptr,
            batch_size, in_channels, in_height, in_width,
            out_channels, kernel_height, kernel_width,
            out_height, out_width, stride, padding, dilation, groups
        );
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive CUDA 2D Convolution with Shared Memory");
}