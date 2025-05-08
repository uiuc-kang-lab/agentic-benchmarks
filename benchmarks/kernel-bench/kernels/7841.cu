#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels, 
    const int in_height,
    const int in_width,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {
    
    __shared__ float input_tile[TILE_HEIGHT * 2][TILE_WIDTH * 2];  // Increased size to handle stride
    __shared__ float weight_tile[8][8];  // Fixed size for typical kernel dimensions
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int h_out_start = by * TILE_HEIGHT;
    const int w_out_start = bx * TILE_WIDTH;
    const int batch_idx = bz / out_channels;
    const int c_out = bz % out_channels;
    
    float sum = bias ? bias[c_out] : 0.0f;
    
    const int input_tile_height = TILE_HEIGHT * stride + kernel_height - 1;
    const int input_tile_width = TILE_WIDTH * stride + kernel_width - 1;
    
    for (int c_in = 0; c_in < in_channels; c_in++) {
        // Load input tile
        for (int i = ty; i < input_tile_height; i += blockDim.y) {
            for (int j = tx; j < input_tile_width; j += blockDim.x) {
                int h_in = h_out_start * stride - padding + i;
                int w_in = w_out_start * stride - padding + j;
                
                float val = 0.0f;
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    val = input[((batch_idx * in_channels + c_in) * in_height + h_in) * in_width + w_in];
                }
                input_tile[i][j] = val;
            }
        }
        
        // Load weight tile with one thread per element for reduced overhead
        if (ty < kernel_height && tx < kernel_width) {
            weight_tile[ty][tx] = weight[((c_out * in_channels + c_in) * kernel_height + ty) * kernel_width + tx];
        }
        
        __syncthreads();
        
        if (ty < TILE_HEIGHT && tx < TILE_WIDTH) {
            int h_out = h_out_start + ty;
            int w_out = w_out_start + tx;
            
            if (h_out < out_height && w_out < out_width) {
                for (int kh = 0; kh < kernel_height; kh++) {
                    for (int kw = 0; kw < kernel_width; kw++) {
                        sum += input_tile[ty * stride + kh][tx * stride + kw] * 
                               weight_tile[kh][kw];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    if (ty < TILE_HEIGHT && tx < TILE_WIDTH) {
        int h_out = h_out_start + ty;
        int w_out = w_out_start + tx;
        
        if (h_out < out_height && w_out < out_width) {
            int out_idx = (batch_idx * out_channels + c_out) * out_height * out_width + 
                         h_out * out_width + w_out;
            output[out_idx] = sum;
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
    
    if (dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias.has_value() ? bias.value() : torch::Tensor(),
                           {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
    }
    
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);
    
    const auto out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    const auto out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, out_height, out_width},
                              x.options());
    
    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocks((out_width + TILE_WIDTH - 1) / TILE_WIDTH,
                (out_height + TILE_HEIGHT - 1) / TILE_HEIGHT,
                batch_size * out_channels);
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_height, kernel_width,
        out_height, out_width,
        stride, padding);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution");
}