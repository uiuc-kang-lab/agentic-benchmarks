#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define MAX_STREAMS 4

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int out_height,
    const int out_width) {

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < out_height && col < out_width && channel < out_channels) {
        float sum = 0.0f;
        
        #pragma unroll 4
        for (int kc = 0; kc < in_channels; ++kc) {
            #pragma unroll
            for (int kh = 0; kh < kernel_size; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int h = row * stride - padding + kh * dilation;
                    const int w = col * stride - padding + kw * dilation;
                    
                    if (h >= 0 && h < in_height && w >= 0 && w < in_width) {
                        const int input_idx = (kc * in_height + h) * in_width + w;
                        const int weight_idx = ((channel * in_channels + kc) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        const int output_idx = (channel * out_height + row) * out_width + col;
        output[output_idx] = sum;
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
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(0);
    
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    cudaStream_t streams[MAX_STREAMS];
    for (int i = 0; i < MAX_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 block(16, 16);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        out_channels);

    const size_t input_slice_size = in_channels * in_height * in_width * sizeof(float);
    const size_t output_slice_size = out_channels * out_height * out_width * sizeof(float);
    
    for (int b = 0; b < batch_size; ++b) {
        const int stream_idx = b % MAX_STREAMS;
        
        conv2d_kernel<<<grid, block, 0, streams[stream_idx]>>>(
            x[b].data_ptr<float>(),
            weight.data_ptr<float>(),
            output[b].data_ptr<float>(),
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            out_height,
            out_width);
        
        if (bias.has_value()) {
            auto bias_view = bias.value().view({1, out_channels, 1, 1});
            output[b].add_(bias_view, 1.0);
        }
    }

    for (int i = 0; i < MAX_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stream-pipelined 2D convolution");
}