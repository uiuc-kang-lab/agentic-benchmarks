#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This CUDA kernel performs 2D convolution on asymmetric input with a square kernel.
// It uses #pragma unroll on the nested loops that iterate over the kernel dimensions to
// reduce loop overhead, which is especially effective when kernel_size is small.

__global__ void conv2d_forward_kernel(
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

    // Identify which output element this thread is responsible for.
    int n = blockIdx.x;       // batch index
    int oc = blockIdx.y;      // output channel index
    int pixel_idx = blockIdx.z * blockDim.x + threadIdx.x;
    if (pixel_idx >= out_height * out_width) return;

    int out_y = pixel_idx / out_width;
    int out_x = pixel_idx % out_width;

    float sum = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Manually unroll kernel height and width loops
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;

                // Boundary check
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = n * in_channels * in_height * in_width
                                  + ic * in_height * in_width
                                  + in_y * in_width
                                  + in_x;
                    int weight_idx = oc * in_channels * kernel_size * kernel_size
                                   + ic * kernel_size * kernel_size
                                   + ky * kernel_size
                                   + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias if provided
    if (bias) {
        sum += bias[oc];
    }
    
    int output_idx = n * out_channels * out_height * out_width
                   + oc * out_height * out_width
                   + out_y * out_width
                   + out_x;
    output[output_idx] = sum;
}


// Host function that prepares the tensors and launches the CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Input checks
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    // This implementation supports groups==1 only
    TORCH_CHECK(groups == 1, "Only groups==1 is supported in this optimized kernel");

    // Extract dimensions
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);

    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2); // square kernel assumed (weight.size(2) == weight.size(3))

    // Compute output dimensions
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    // Determine kernel launch parameters
    const int output_pixels = out_height * out_width;
    const int threads = 256;
    const int blocks_z = (output_pixels + threads - 1) / threads;
    dim3 grid(batch, out_channels, blocks_z);
    dim3 block(threads);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = (bias.has_value()) ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // Launch the CUDA convolution kernel
    conv2d_forward_kernel<<<grid, block>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, dilation);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA forward function for 2D convolution with manual loop unrolling");
}
