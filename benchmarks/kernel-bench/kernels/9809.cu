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
    // Calculate position in output tensor
    int ow = threadIdx.x + blockIdx.x * blockDim.x;
    int oh = (blockIdx.y / channels) % out_h;
    int c = blockIdx.y / out_h;
    int b = blockIdx.z;

    if (ow < out_w && c < channels && b < batch) {
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

    // Set up CUDA launch parameters with streams
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 threads(128, 1, 1);  // Adjusted block size for experimentation
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        channels * out_h,
        batch
    );

    float* input_d;
    float* weight_d;
    float* bias_d;
    float* output_d;
    cudaMalloc((void**)&input_d, x.numel() * sizeof(float));
    cudaMalloc((void**)&weight_d, weight.numel() * sizeof(float));
    cudaMalloc((void**)&bias_d, bias_val.numel() * sizeof(float));
    cudaMalloc((void**)&output_d, output.numel() * sizeof(float));

    cudaMemcpyAsync(input_d, x.data_ptr<float>(), x.numel() * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(weight_d, weight.data_ptr<float>(), weight.numel() * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(bias_d, bias_val.data_ptr<float>(), bias_val.numel() * sizeof(float), cudaMemcpyHostToDevice, stream);

    depthwise_conv2d_kernel<<<blocks, threads, 0, stream>>>(
        input_d,
        weight_d,
        bias_d,
        output_d,
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

    cudaMemcpyAsync(output.data_ptr<float>(), output_d, output.numel() * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaFree(input_d);
    cudaFree(weight_d);
    cudaFree(bias_d);
    cudaFree(output_d);
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution forward with Streams (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}