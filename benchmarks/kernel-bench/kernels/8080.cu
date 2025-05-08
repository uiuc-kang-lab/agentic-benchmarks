#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function to handle bias addition
__device__ void apply_bias(float* output, const float* bias, 
                         int channels, int output_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_length * channels) {
        int c = idx % channels;
        output[idx] += bias[c];
    }
}

// Device function for main convolution computation
__global__ void compute_conv_transpose(
    const float* input, const float* weight,
    float* output, int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    int output_padding, int input_length) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_length = (input_length - 1) * stride - 2 * padding + 
                       kernel_size + output_padding;
    
    if (idx < output_length * out_channels) {
        int out_pos = idx / out_channels;
        int out_ch = idx % out_channels;
        float sum = 0.0f;
        
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            for (int k = 0; k < kernel_size; ++k) {
                int in_pos = (out_pos + padding - k) / stride;
                if (in_pos >= 0 && in_pos < input_length && 
                    (out_pos + padding - k) % stride == 0) {
                    sum += input[in_pos * in_channels + in_ch] * 
                           weight[out_ch * in_channels * kernel_size + 
                                  in_ch * kernel_size + k];
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    int threads = 256;
    int blocks = (output_length * out_channels + threads - 1) / threads;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int b = 0; b < batch_size; ++b) {
        compute_conv_transpose<<<blocks, threads, 0, stream>>>(
            x[b].data_ptr<float>(), weight.data_ptr<float>(),
            output[b].data_ptr<float>(), in_channels, out_channels,
            kernel_size, stride, padding, output_padding, input_length);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        auto bias_ptr = bias_tensor.data_ptr<float>();
        for (int b = 0; b < batch_size; ++b) {
            apply_bias<<<blocks, threads, 0, stream>>>(
                output[b].data_ptr<float>(), bias_ptr, out_channels, output_length);
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA)");
}
