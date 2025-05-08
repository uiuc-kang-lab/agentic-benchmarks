#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
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
    const int kernel_height,
    const int kernel_width,
    const int stride,
    const int padding) {
    
    const int out_height = (height + 2 * padding - kernel_height) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_width) / stride + 1;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    const int n = bid / (out_channels * out_height * out_width);
    const int rem = bid % (out_channels * out_height * out_width);
    const int oc = rem / (out_height * out_width);
    const int h = (rem / out_width) % out_height;
    const int w = rem % out_width;
    
    if (n >= batch_size) return;
    
    scalar_t sum = 0;
    
    #pragma unroll
    for (int ic = lane; ic < in_channels; ic += WARP_SIZE) {
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
                const int h_in = h * stride - padding + kh;
                const int w_in = w * stride - padding + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    const scalar_t val = input[
                        ((n * in_channels + ic) * height + h_in) * width + w_in];
                    const scalar_t wt = weight[
                        ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw];
                    sum += val * wt;
                }
            }
        }
    }
    
    // Warp reduction using shuffle operations
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) {
        output[((n * out_channels + oc) * out_height + h) * out_width + w] = sum;
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
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);
    
    const int out_height = (height + 2 * padding - kernel_height) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_width) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              x.options());
    
    const int threads = 256;
    const int blocks = batch_size * out_channels * out_height * out_width;
    
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
            kernel_height,
            kernel_width,
            stride,
            padding);
    }));
    
    if (bias.has_value()) {
        output += bias.value().view({1, -1, 1, 1});
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution");
}