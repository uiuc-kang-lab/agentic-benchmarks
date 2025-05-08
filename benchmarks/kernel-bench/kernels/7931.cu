#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    const int output_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    __shared__ float shared_input[BLOCK_SIZE + 2 * (BLOCK_SIZE - 1)][BLOCK_SIZE + 2 * (BLOCK_SIZE - 1)];
    __shared__ float shared_weight[BLOCK_SIZE][BLOCK_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_SIZE;
    const int by = blockIdx.y * BLOCK_SIZE;
    const int out_x = bx + tx;
    const int out_y = by + ty;
    
    // Loop over output channels
    for (int oc = 0; oc < out_channels; ++oc) {
        float sum = 0.0f;
        
        // Loop over input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            __syncthreads();
            
            // Load input tile with padding
            for (int i = ty; i < BLOCK_SIZE + 2 * (kernel_size - 1); i += BLOCK_SIZE) {
                for (int j = tx; j < BLOCK_SIZE + 2 * (kernel_size - 1); j += BLOCK_SIZE) {
                    int y_im = by * stride + i - padding;
                    int x_im = bx * stride + j - padding;
                    
                    if (y_im >= 0 && y_im < height && x_im >= 0 && x_im < width) {
                        shared_input[i][j] = input[ic * height * width + y_im * width + x_im];
                    } else {
                        shared_input[i][j] = 0.0f;
                    }
                }
            }
            
            // Load weights
            if (tx < kernel_size && ty < kernel_size) {
                shared_weight[ty][tx] = weight[
                    oc * in_channels * kernel_size * kernel_size +
                    ic * kernel_size * kernel_size +
                    ty * kernel_size + tx];
            }
            
            __syncthreads();
            
            // Compute convolution
            if (out_x < output_width && out_y < output_height) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int in_y = ty * stride + ky;
                        int in_x = tx * stride + kx;
                        sum += shared_input[in_y][in_x] * shared_weight[ky][kx];
                    }
                }
            }
        }
        
        // Write output
        if (out_x < output_width && out_y < output_height) {
            output[oc * output_height * output_width + out_y * output_width + out_x] = sum;
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
    
    auto input_size = x.sizes();
    auto weight_size = weight.sizes();
    
    int batch_size = input_size[0];
    int in_channels = input_size[1];
    int height = input_size[2];
    int width = input_size[3];
    int out_channels = weight_size[0];
    int kernel_size = weight_size[2];
    
    auto output = torch::zeros({batch_size, out_channels,
                              (height + 2*padding - kernel_size) / stride + 1,
                              (width + 2*padding - kernel_size) / stride + 1},
                              x.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding);
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with optional bias");
}