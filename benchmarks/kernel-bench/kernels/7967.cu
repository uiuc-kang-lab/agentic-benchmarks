#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void conv2d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    __shared__ scalar_t shared_input[TILE_SIZE + 2][TILE_SIZE + 2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int b = blockIdx.z / out_channels;
    int oc = blockIdx.z % out_channels;
    
    int h_out = (height + 2 * padding - kernel_size) / stride + 1;
    int w_out = (width + 2 * padding - kernel_size) / stride + 1;
    
    // Load input tile into shared memory
    for (int ic = 0; ic < in_channels; ic++) {
        if (by + ty < height && bx + tx < width) {
            int in_idx = ((b * in_channels + ic) * height + by + ty) * width + bx + tx;
            shared_input[ty][tx] = input[in_idx];
        } else {
            shared_input[ty][tx] = 0;
        }
        __syncthreads();
        
        // Compute convolution for this input channel
        if (ty < TILE_SIZE && tx < TILE_SIZE) {
            scalar_t sum = 0;
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h_in = by + ty * stride - padding + kh;
                    int w_in = bx + tx * stride - padding + kw;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                        sum += shared_input[ty + kh][tx + kw] * weight[weight_idx];
                    }
                }
            }
            
            if (by + ty < h_out && bx + tx < w_out) {
                int out_idx = ((b * out_channels + oc) * h_out + by + ty) * w_out + bx + tx;
                output[out_idx] += sum;
            }
        }
        __syncthreads();
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
    auto kernel_size = weight.size(2);
    
    auto h_out = (height + 2 * padding - kernel_size) / stride + 1;
    auto w_out = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((w_out + TILE_SIZE - 1) / TILE_SIZE,
                (h_out + TILE_SIZE - 1) / TILE_SIZE,
                batch_size * out_channels);
    
    AT_DISPATCH_FLOATING_TYPES(x.type(), "conv2d_forward_cuda", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_size,
            stride,
            padding);
    }));
    
    if (bias.has_value()) {
        output += bias.value().view({1, out_channels, 1, 1});
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with optional bias");
}