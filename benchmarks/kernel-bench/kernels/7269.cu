#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16  // Using smaller tile size for better occupancy
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
    
    __shared__ float shared_input[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int b = blockIdx.z / out_channels;
    int oc = blockIdx.z % out_channels;
    
    float sum = bias ? bias[oc] : 0.0f;
    
    int start_h = by + ty - padding;
    int start_w = bx + tx - padding;

    for (int ic = 0; ic < in_channels; ic++) {
        float input_val = 0.0f;
        
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            int h = start_h + kh;
            bool h_in_bounds = (h >= 0) && (h < in_height);

            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                int w = start_w + kw;
                bool w_in_bounds = (w >= 0) && (w < in_width);

                if (h_in_bounds && w_in_bounds) {
                    input_val = input[((b * in_channels + ic) * in_height + h) * in_width + w];
                } else {
                    input_val = 0.0f;
                }

                float weight_val = weight[((oc * in_channels + ic) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw];
                sum += input_val * weight_val;
            }
        }
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
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    conv2d_kernel<<<blocks, threads>>>(
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
