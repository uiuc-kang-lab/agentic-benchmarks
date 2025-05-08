#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Constant memory for weights (64KB limit)
__constant__ float const_weight[16384]; // 64KB / 4 bytes = 16384 float elements
__constant__ float const_bias[1024];    // Space for bias values

__global__ void conv2d_const_mem_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const bool has_bias) {

    const int n = blockIdx.x;
    const int oc = blockIdx.y;
    const int out_y = blockIdx.z * blockDim.y + threadIdx.y;
    const int out_x = threadIdx.x;

    if (out_y >= out_height || out_x >= out_width) return;

    float sum = 0.0f;
    
    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int in_y = out_y * stride - padding + ky * dilation;
                const int in_x = out_x * stride - padding + kx * dilation;

                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    const int input_idx = n * in_channels * in_height * in_width +
                                        ic * in_height * in_width +
                                        in_y * in_width + in_x;
                    
                    const int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                         ic * kernel_size * kernel_size +
                                         ky * kernel_size + kx;
                    
                    sum += input[input_idx] * const_weight[weight_idx];
                }
            }
        }
    }

    if (has_bias) {
        sum += const_bias[oc];
    }

    const int output_idx = n * out_channels * out_height * out_width +
                          oc * out_height * out_width +
                          out_y * out_width + out_x;
    output[output_idx] = sum;
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

    TORCH_CHECK(groups == 1, "Only groups==1 is supported");

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    // Check if weight tensor fits in constant memory
    TORCH_CHECK(weight.numel() <= 16384, "Weight tensor too large for constant memory");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().numel() <= 1024, "Bias tensor too large for constant memory");
    }

    // Copy weight tensor to constant memory
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float));
    
    if (bias.has_value()) {
        cudaMemcpyToSymbol(const_bias, bias.value().data_ptr<float>(), 
                          bias.value().numel() * sizeof(float));
    }

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    dim3 threads(32, 8);  // 256 threads per block
    dim3 blocks(batch, 
                out_channels, 
                (out_height + threads.y - 1) / threads.y);

    conv2d_const_mem_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, dilation,
        bias.has_value());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Constant memory optimized CUDA convolution implementation");
}