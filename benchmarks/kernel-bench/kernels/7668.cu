#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized block dimensions for H100
#define BLOCK_D 4
#define BLOCK_H 8
#define BLOCK_W 8
#define TILE_SIZE 8

__global__ void conv3d_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation,
    const int groups) {

    __shared__ float shared_input[TILE_SIZE][TILE_SIZE][TILE_SIZE];
    __shared__ float shared_weight[TILE_SIZE][TILE_SIZE][TILE_SIZE];

    // 3D thread indexing
    const int d_out = blockIdx.z * BLOCK_D + threadIdx.z;
    const int h_out = blockIdx.y * BLOCK_H + threadIdx.y;
    const int w_out = blockIdx.x * BLOCK_W + threadIdx.x;
    
    // Early exit if outside output bounds
    if (d_out >= out_depth || h_out >= out_height || w_out >= out_width) {
        return;
    }

    const int channels_per_group = out_channels / groups;
    
    // Loop over batches and output channels
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            const int group = oc / channels_per_group;
            const int in_channels_per_group = in_channels / groups;
            float sum = 0.0f;

            // Compute input position
            const int d_in_start = d_out * stride - padding;
            const int h_in_start = h_out * stride - padding;
            const int w_in_start = w_out * stride - padding;

            // Loop over input channels in group
            for (int ic = 0; ic < in_channels_per_group; ic++) {
                const int in_c = group * in_channels_per_group + ic;

                // Convolution computation with shared memory tiles
                #pragma unroll
                for (int kd = 0; kd < kernel_d; kd++) {
                    const int d_in = d_in_start + kd * dilation;
                    if (d_in >= 0 && d_in < in_depth) {
                        #pragma unroll
                        for (int kh = 0; kh < kernel_h; kh++) {
                            const int h_in = h_in_start + kh * dilation;
                            if (h_in >= 0 && h_in < in_height) {
                                #pragma unroll
                                for (int kw = 0; kw < kernel_w; kw++) {
                                    const int w_in = w_in_start + kw * dilation;
                                    if (w_in >= 0 && w_in < in_width) {
                                        const float input_val = input[
                                            ((b * in_channels + in_c) * in_depth + d_in) * 
                                            in_height * in_width + h_in * in_width + w_in
                                        ];
                                        const float weight_val = weight[
                                            ((oc * in_channels_per_group + ic) * kernel_d + kd) * 
                                            kernel_h * kernel_w + kh * kernel_w + kw
                                        ];
                                        sum += input_val * weight_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Add bias if present
            if (bias != nullptr) {
                sum += bias[oc];
            }

            // Write output
            const int out_idx = ((b * out_channels + oc) * out_depth + d_out) * 
                               out_height * out_width + h_out * out_width + w_out;
            output[out_idx] = sum;
        }
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

    auto bias = bias_opt.value_or(at::Tensor());
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    // Calculate output dimensions
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    // Create output tensor
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    // Calculate grid dimensions
    dim3 threads(BLOCK_W, BLOCK_H, BLOCK_D);
    dim3 blocks(
        (out_width + BLOCK_W - 1) / BLOCK_W,
        (out_height + BLOCK_H - 1) / BLOCK_H,
        (out_depth + BLOCK_D - 1) / BLOCK_D
    );

    // Launch kernel
    conv3d_tiled_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation, groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with tiled indexing (CUDA)");
}