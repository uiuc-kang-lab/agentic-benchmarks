#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 32

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
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
    const int dilation) {

    // 2D thread block mapping
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z / out_channels;  // batch index
    const int oc = blockIdx.z % out_channels; // output channel

    // Early exit if thread is out of bounds
    if (out_x >= out_width || out_y >= out_height || n >= batch) {
        return;
    }

    float sum = 0.0f;

    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            const int in_y = out_y * stride - padding + ky * dilation;
            
            // Skip if outside vertical bounds
            if (in_y < 0 || in_y >= in_height) continue;

            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int in_x = out_x * stride - padding + kx * dilation;
                
                // Skip if outside horizontal bounds
                if (in_x < 0 || in_x >= in_width) continue;

                const int input_idx = ((n * in_channels + ic) * in_height + in_y) * in_width + in_x;
                const int weight_idx = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    const int output_idx = ((n * out_channels + oc) * out_height + out_y) * out_width + out_x;
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

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    // Calculate grid dimensions for 2D block mapping
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch * out_channels
    );

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv2d_kernel<<<blocks, threads>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized thread mapping CUDA convolution implementation");
}