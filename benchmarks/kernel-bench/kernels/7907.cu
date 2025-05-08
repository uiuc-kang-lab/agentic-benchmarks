#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses stride loops to cover the workload in the spatial dimensions, allowing threads to
// process multiple output elements when the output tensor is larger than the total number of threads.
// It supports padding, stride, and dilation (assuming a square kernel) while computing the correct result.

__global__ void conv2d_stride_loop_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int output_height,
    int output_width,
    int stride,
    int padding,
    int dilation) {

    // Each block in z dimension corresponds to one (batch, out_channel) pair
    int batch_channel = blockIdx.z;
    int n = batch_channel / out_channels;
    int oc = batch_channel % out_channels;

    // Use 2D stride loops to cover the output spatial dimensions
    for (int oy = blockIdx.y * blockDim.y + threadIdx.y; oy < output_height; oy += gridDim.y * blockDim.y) {
        for (int ox = blockIdx.x * blockDim.x + threadIdx.x; ox < output_width; ox += gridDim.x * blockDim.x) {
            float sum = 0.0f;
            
            // Loop over input channels and kernel window
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        // Incorporate dilation in the computation of the input indices
                        int in_y = oy * stride - padding + kh * dilation;
                        int in_x = ox * stride - padding + kw * dilation;
                        
                        if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                            int input_idx = ((n * in_channels + ic) * input_height + in_y) * input_width + in_x;
                            int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            if (bias != nullptr) {
                sum += bias[oc];
            }
            
            int out_idx = ((n * out_channels + oc) * output_height + oy) * output_width + ox;
            output[out_idx] = sum;
        }
    }
}

// The forward function computes the output dimensions using stride, padding, and dilation. It then
// launches the kernel with a 3D grid where the z-dimension indexes the (batch, out_channel) pairs,
// and the x and y dimensions are processed using stride loops to cover the entire output spatial area.

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
    TORCH_CHECK(groups == 1, "groups != 1 not supported by this custom kernel");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // assuming square kernel

    // Compute output dimensions considering dilation
    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width  = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, x.options());
    
    // Define block dimensions and use stride loops in the kernel for spatial dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (output_width + blockDim.x - 1) / blockDim.x,
        (output_height + blockDim.y - 1) / blockDim.y,
        batch_size * out_channels  // each z index corresponds to one (batch, out_channel) pair
    );

    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    conv2d_stride_loop_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        output_height,
        output_width,
        stride,
        padding,
        dilation);

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with stride loops");
}
