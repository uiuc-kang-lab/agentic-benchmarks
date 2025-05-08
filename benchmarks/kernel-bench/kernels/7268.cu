#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
    
    extern __shared__ float shared_input[];
    float* tile_input = shared_input;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tile_width = blockDim.x + KERNEL_SIZE - 1;
    int tile_height = blockDim.y + KERNEL_SIZE - 1;
    int input_tile_row = blockIdx.y * blockDim.y - padding;
    int input_tile_col = blockIdx.x * blockDim.x - padding;
    int b = blockIdx.z / out_channels;
    int oc = blockIdx.z % out_channels;
    
    float sum = bias ? bias[oc] : 0.0f;
    
    int start_h = by + ty - padding;
    int start_w = bx + tx - padding;

    for (int ic = 0; ic < in_channels; ic++) {
        if (start_h >= 0 && start_h < in_height && start_w >= 0 && start_w < in_width) {
            tile_input[ty * blockDim.x + tx] = input[((b * in_channels + ic) * in_height + start_h) * in_width + start_w];
        } else {
            tile_input[ty * blockDim.x + tx] = 0.0f;
        }
        __syncthreads();

        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                int h = start_h + kh;
                int w = start_w + kw;

                if (h >= 0 && h < in_height && w >= 0 && w < in_width) {
                    float input_val = tile_input[(kh+ty) * blockDim.x + (kw+tx)];
                    float weight_val = weight[((oc * in_channels + ic) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw];
                    sum += input_val * weight_val;
                }
            }
        }
        __syncthreads();
    }
    
    int out_h = by + ty;
    int out_w = bx + tx;
    if (out_h < out_height && out_w < out_width) {
        output[((b * out_channels + oc) * out_height + out_h) * out_width + out_w] = sum;
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
    
    int block_size = 32; // Adjusted block size for potential performance improvement
    dim3 threads(block_size, block_size);
    dim3 blocks(
        (out_width + block_size - 1) / block_size,
        (out_height + block_size - 1) / block_size,
        batch_size * out_channels
    );
    
    size_t shared_memory_size = sizeof(float) * (block_size + KERNEL_SIZE - 1) * (block_size + KERNEL_SIZE - 1);

    conv2d_kernel<<<blocks, threads, shared_memory_size>>>(
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