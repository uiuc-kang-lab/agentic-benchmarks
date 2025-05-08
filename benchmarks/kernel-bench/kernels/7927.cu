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
    
    __shared__ float shared_input[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ float shared_weight[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int x = bx + tx;
    int y = by + ty;
    
    float sum = 0.0f;
    
    for (int c = 0; c < in_channels; ++c) {
        if (x < width && y < height) {
            shared_input[ty][tx] = input[c * height * width + y * width + x];
        }
        
        if (tx < kernel_size && ty < kernel_size) {
            shared_weight[ty][tx] = weight[c * kernel_size * kernel_size + ty * kernel_size + tx];
        }
        
        __syncthreads();
        
        if (x < width && y < height) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int ix = tx + kx - padding;
                    int iy = ty + ky - padding;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        sum += shared_input[iy][ix] * shared_weight[ky][kx];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (tx == 0 && x < width && y < height) {
        output[y * width + x] = sum;
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