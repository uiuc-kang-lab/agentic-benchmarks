#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

__global__ void depthwise_conv2d_stride_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int stride,
    int padding,
    int dilation) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int total_elements = batch * channels * out_h * out_w;

    for (int idx = index; idx < total_elements; idx += total_threads) {
        int ow = idx % out_w;
        int tmp = idx / out_w;
        int oh = tmp % out_h;
        tmp = tmp / out_h;
        int c = tmp % channels;
        int b = tmp / channels;

        if (b < batch) {
            float sum = 0.0f;
            
            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh) {
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                    int weight_idx = c * kernel_h + kh;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
            
            sum += bias[c];
            int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
            output[output_idx] = sum;
        }
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    x = x.contiguous();
    weight = weight.contiguous();
    
    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);
    
    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }
    
    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }
    
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;
    
    auto output = at::empty({batch, channels, out_h, out_w}, x.options());
    
    int total_elements = batch * channels * out_h * out_w;
    int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias_val.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    depthwise_conv2d_stride_kernel<<<blocks, BLOCK_SIZE>>>(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        stride,
        padding,
        dilation
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}