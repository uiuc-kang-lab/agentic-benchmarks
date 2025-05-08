#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void depthwise_conv2d_kernel(
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
    int dilation)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (index < total) {
        int ow = index % out_w;
        int tmp = index / out_w;
        int oh = tmp % out_h;
        tmp = tmp / out_h;
        int c = tmp % channels;
        int b = tmp / channels;

        float sum = 0.f;
        int iw = ow * stride - padding;

        // Specialized paths for common kernel heights
        if (kernel_h == 3) {
            // Unrolled loop for kernel_h = 3
            int ih0 = oh * stride - padding;
            int ih1 = ih0 + dilation;
            int ih2 = ih1 + dilation;
            
            if (ih0 >= 0 && ih0 < in_h && iw >= 0 && iw < in_w)
                sum += input[((b * channels + c) * in_h + ih0) * in_w + iw] * weight[c * kernel_h + 0];
            if (ih1 >= 0 && ih1 < in_h && iw >= 0 && iw < in_w)
                sum += input[((b * channels + c) * in_h + ih1) * in_w + iw] * weight[c * kernel_h + 1];
            if (ih2 >= 0 && ih2 < in_h && iw >= 0 && iw < in_w)
                sum += input[((b * channels + c) * in_h + ih2) * in_w + iw] * weight[c * kernel_h + 2];
        }
        else if (kernel_h == 5) {
            // Unrolled loop for kernel_h = 5
            int ih0 = oh * stride - padding;
            int ih1 = ih0 + dilation;
            int ih2 = ih1 + dilation;
            int ih3 = ih2 + dilation;
            int ih4 = ih3 + dilation;
            
            if (ih0 >= 0 && ih0 < in_h && iw >= 0 && iw < in_w)
                sum += input[((b * channels + c) * in_h + ih0) * in_w + iw] * weight[c * kernel_h + 0];
            if (ih1 >= 0 && ih1 < in_h && iw >= 0 && iw < in_w)
                sum += input[((b * channels + c) * in_h + ih1) * in_w + iw] * weight[c * kernel_h + 1];
            if (ih2 >= 0 && ih2 < in_h && iw >= 0 && iw < in_w)
                sum += input[((b * channels + c) * in_h + ih2) * in_w + iw] * weight[c * kernel_h + 2];
            if (ih3 >= 0 && ih3 < in_h && iw >= 0 && iw < in_w)
                sum += input[((b * channels + c) * in_h + ih3) * in_w + iw] * weight[c * kernel_h + 3];
            if (ih4 >= 0 && ih4 < in_h && iw >= 0 && iw < in_w)
                sum += input[((b * channels + c) * in_h + ih4) * in_w + iw] * weight[c * kernel_h + 4];
        }
        else {
            // Fallback for other kernel heights with pragma unroll hint
            #pragma unroll 7
            for (int kh = 0; kh < kernel_h; ++kh) {
                int ih = oh * stride - padding + kh * dilation;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                    int weight_idx = c * kernel_h + kh;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        
        sum += bias[c];
        output[((b * channels + c) * out_h + oh) * out_w + ow] = sum;
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups)
{
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

    int total = batch * channels * out_h * out_w;
    int threads = 1024;
    int blocks = (total + threads - 1) / threads;

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_val.data_ptr<float>(),
        output.data_ptr<float>(),
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