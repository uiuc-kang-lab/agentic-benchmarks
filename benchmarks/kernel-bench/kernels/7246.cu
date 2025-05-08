#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(
float* output, const float* input, const float* weight,
const float* bias, int batch_size, int in_channels,
int in_height, int in_width, int out_channels,
int kernel_size, int stride, int padding) {
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + TILE_SIZE * TILE_SIZE;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    int out_h = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    int h_out = by * TILE_SIZE + ty;
    int w_out = bx * TILE_SIZE + tx;
    
    if (h_out >= out_h || w_out >= out_w) return;
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    float input_val = input[((bz * in_channels + ic) * in_height + h_in) * in_width + w_in];
                    float weight_val = weight[((bz * out_channels + ic) * kernel_size + kh) * kernel_size + kw];
                    sum += input_val * weight_val;
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[bz];
    }
    
    output[(bz * out_h + h_out) * out_w + w_out] = sum;
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
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((out_width + TILE_SIZE - 1) / TILE_SIZE,
                (out_height + TILE_SIZE - 1) / TILE_SIZE,
                batch_size * out_channels);
    
    int shared_mem_size = TILE_SIZE * TILE_SIZE * sizeof(float) * 2;
    
    conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size, stride, padding);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution");
}