#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel using grid-stride loops to handle workloads larger than the number of threads
__global__ void conv2d_stride_loop_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Total number of output elements
    int total = batch * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_amount = blockDim.x * gridDim.x;
    
    // Grid-stride loop to cover all output elements
    for (int index = idx; index < total; index += stride_amount) {
        // Decode the linear index into (n, oc, i, j)
        int j = index % out_width;
        int tmp = index / out_width;
        int i = tmp % out_height;
        tmp = tmp / out_height;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;

        float sum = 0.0f;
        
        // Convolution over input channels and the kernel window
        for (int ic = 0; ic < in_channels; ic++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = i * stride - padding + ky * dilation;
                    int in_x = j * stride - padding + kx * dilation;
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        int input_idx = n * in_channels * in_height * in_width
                                      + ic * in_height * in_width
                                      + in_y * in_width + in_x;
                        int weight_idx = oc * in_channels * kernel_size * kernel_size
                                       + ic * kernel_size * kernel_size
                                       + ky * kernel_size + kx;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[oc];
        }

        output[index] = sum;
    }
}

// Host function that sets up the kernel launch parameters and calls the CUDA kernel
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

    // For simplicity, this kernel supports groups == 1 only
    TORCH_CHECK(groups == 1, "Only groups==1 is supported in conv2d_stride_loop_base");

    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width  = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // assumed square kernel

    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width  = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    int total = batch * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = (bias.has_value()) ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv2d_stride_loop_kernel<<<blocks, threads>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, dilation);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution using grid-stride loops");
}
