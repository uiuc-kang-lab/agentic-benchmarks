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
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding) {
    
    __shared__ float partial_sum[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    int out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    int h_out = by * TILE_SIZE + ty;
    int w_out = bx * TILE_SIZE + tx;
    int c_out = bz;
    
    if (h_out < out_h && w_out < out_w) {
        float sum = 0.0f;
        
        for (int n = 0; n < batch_size; n++) {
            for (int c = 0; c < in_channels; c++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int h_in = h_out * stride - padding + kh;
                        int w_in = w_out * stride - padding + kw;
                        
                        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                            int in_idx = ((n * in_channels + c) * height + h_in) * width + w_in;
                            int w_idx = ((c_out * in_channels + c) * kernel_h + kh) * kernel_w + kw;
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        
        partial_sum[ty][tx] = sum;
        __syncthreads();
        
        // Reduce within thread block
        if (ty == 0 && tx == 0) {
            float block_sum = 0.0f;
            for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
                    if ((by * TILE_SIZE + i < out_h) && (bx * TILE_SIZE + j < out_w)) {
                        block_sum += partial_sum[i][j];
                    }
                }
            }
            
            // Only one atomic add per block
            int out_idx = (c_out * out_h + h_out) * out_w + w_out;
            if (bias != nullptr) {
                atomicAdd(&output[out_idx], block_sum + bias[c_out]);
            } else {
                atomicAdd(&output[out_idx], block_sum);
            }
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
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    auto out_h = (height + 2 * padding - kernel_h) / stride + 1;
    auto out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_w + TILE_SIZE - 1) / TILE_SIZE,
        (out_h + TILE_SIZE - 1) / TILE_SIZE,
        out_channels
    );
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride,
        padding
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution");
}