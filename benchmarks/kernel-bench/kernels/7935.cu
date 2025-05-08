#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_kernel = shared_mem + TILE_SIZE * TILE_SIZE;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int batch_idx = blockIdx.z / out_channels;
    int out_ch = blockIdx.z % out_channels;
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic++) {
        // Load input tile to shared memory
        if (tx < TILE_SIZE && ty < TILE_SIZE) {
            int x = bx + tx - padding;
            int y = by + ty - padding;
            if (x >= 0 && x < width && y >= 0 && y < height) {
                shared_input[ty * TILE_SIZE + tx] = input[
                    batch_idx * (in_channels * height * width) +
                    ic * (height * width) +
                    y * width + x];
            } else {
                shared_input[ty * TILE_SIZE + tx] = 0.0f;
            }
        }
        
        // Load kernel tile to shared memory
        if (tx < kernel_size && ty < kernel_size) {
            shared_kernel[ty * kernel_size + tx] = kernel[
                out_ch * (in_channels * kernel_size * kernel_size) +
                ic * (kernel_size * kernel_size) +
                ty * kernel_size + tx];
        }
        
        __syncthreads();
        
        // Compute convolution
        if (tx < TILE_SIZE && ty < TILE_SIZE) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    sum += shared_input[(ty + ky) * TILE_SIZE + (tx + kx)] *
                           shared_kernel[ky * kernel_size + kx];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    if (tx < TILE_SIZE && ty < TILE_SIZE) {
        int out_x = bx + tx;
        int out_y = by + ty;
        if (out_x < width && out_y < height) {
            output[batch_idx * (out_channels * height * width) +
                   out_ch * (height * width) +
                   out_y * width + out_x] = sum;
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
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int height = x.size(2);
    const int width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    auto output = torch::zeros({batch_size, out_channels, height, width},
                              x.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (width + TILE_SIZE - 1) / TILE_SIZE,
        (height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    const int shared_mem_size = (TILE_SIZE * TILE_SIZE + kernel_size * kernel_size) * sizeof(float);
    
    conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
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
        padding
    );
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with optional bias");
}