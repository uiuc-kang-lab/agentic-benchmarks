#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16
#define KERNEL_SIZE 3

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const float* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int stride,
    int padding) {
    
    __shared__ float shared_input[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int b = blockIdx.z / out_channels;
    int oc = blockIdx.z % out_channels;
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic++) {
        // Load input tile to shared memory
        int in_x = bx + tx - padding;
        int in_y = by + ty - padding;
        
        if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
            shared_input[ty][tx] = input[((b * in_channels + ic) * in_height + in_y) * in_width + in_x];
        } else {
            shared_input[ty][tx] = 0.0f;
        }
        __syncthreads();
        
        // Compute convolution
        if (tx < TILE_SIZE && ty < TILE_SIZE) {
            int out_x = bx + tx;
            int out_y = by + ty;
            
            if (out_x < out_width && out_y < out_height) {
                for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                    for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                        float w = weight[((oc * in_channels + ic) * KERNEL_SIZE + ky) * KERNEL_SIZE + kx];
                        sum += w * shared_input[ty * stride + ky][tx * stride + kx];
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Write output
    int out_x = bx + tx;
    int out_y = by + ty;
    
    if (out_x < out_width && out_y < out_height) {
        int out_idx = ((b * out_channels + oc) * out_height + out_y) * out_width + out_x;
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[out_idx] = sum;
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
    auto kernel_size = weight.size(2);
    
    auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              x.options());
    
    dim3 threads(TILE_SIZE + KERNEL_SIZE - 1, TILE_SIZE + KERNEL_SIZE - 1);
    dim3 blocks((out_width + TILE_SIZE - 1) / TILE_SIZE,
                (out_height + TILE_SIZE - 1) / TILE_SIZE,
                batch_size * out_channels);
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        stride,
        padding);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution");
}