#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void conv2d_kernel_shared_memory(
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

    extern __shared__ float shared_mem[];

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z;

    if (w_out >= out_width || h_out >= out_height || c_out >= out_channels) return;

    float sum = bias ? bias[c_out] : 0.0f;

    for (int b = 0; b < batch_size; ++b) {
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = ((b * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_height + kh) * kernel_width + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    shared_mem[threadIdx.y * blockDim.x + threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.y == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < blockDim.y; ++i) {
            block_sum = warpReduceSum(shared_mem[i * blockDim.x + threadIdx.x]);
        }
        if (threadIdx.x == 0) {
            int out_idx = ((b * out_channels + c_out) * out_height + h_out) * out_width + w_out;
            output[out_idx] = block_sum;
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

    dim3 threads(16, 16);
    dim3 blocks((out_width + threads.x - 1) / threads.x, 
                (out_height + threads.y - 1) / threads.y, 
                out_channels);

    size_t shared_memory_size = threads.x * threads.y * sizeof(float);
    
    conv2d_kernel_shared_memory<<<blocks, threads, shared_memory_size>>>(
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