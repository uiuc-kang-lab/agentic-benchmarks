#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized CUDA kernel for 2D convolution that uses __ldg() for read-only global memory loads.
// Memory accesses for input, weight, and bias are assumed to be 128-bit aligned to improve throughput.
// Each thread computes one output pixel by mapping a 1D thread index to the 2D spatial coordinates.

__global__ void conv2d_ldg_aligned_kernel(
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

    // Map each thread to one output pixel
    int n = blockIdx.x;       // batch index
    int oc = blockIdx.y;      // output channel index
    int pixel_index = blockIdx.z * blockDim.x + threadIdx.x;
    int total_pixels = out_height * out_width;
    if (pixel_index >= total_pixels) return;

    // Convert linear index to 2D coordinates
    int out_y = pixel_index / out_width;
    int out_x = pixel_index % out_width;

    float sum = 0.0f;
    
    // Loop over all input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Unroll kernel height and width loops
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;
                
                // Check bounds
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = n * (in_channels * in_height * in_width)
                                  + ic * (in_height * in_width)
                                  + in_y * in_width + in_x;
                    int weight_idx = oc * (in_channels * kernel_size * kernel_size)
                                   + ic * (kernel_size * kernel_size)
                                   + ky * kernel_size + kx;
                    
                    // Load using __ldg() for read-only global memory accesses (assumed 128-bit aligned)
                    float in_val = __ldg(&input[input_idx]);
                    float w_val  = __ldg(&weight[weight_idx]);
                    sum += in_val * w_val;
                }
            }
        }
    }
    
    // Add bias if available
    if (bias) {
        sum += __ldg(&bias[oc]);
    }

    int output_idx = n * (out_channels * out_height * out_width)
                   + oc * (out_height * out_width)
                   + out_y * out_width + out_x;
    output[output_idx] = sum;
}

// Host function to launch the CUDA kernel
// Groups must be 1 in this implementation

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

    TORCH_CHECK(groups == 1, "Only groups==1 is supported in conv2d_ldg_aligned_tb");

    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);  // Assuming square kernel (weight.size(2)==weight.size(3))
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    int total_pixels = out_height * out_width;
    int threads = 256;
    int blocks_z = (total_pixels + threads - 1) / threads;

    // Grid: [batch, out_channels, number_of_blocks_to_cover_output_pixels]
    dim3 grid(batch, out_channels, blocks_z);
    dim3 block(threads);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv2d_ldg_aligned_kernel<<<grid, block>>>(
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
        dilation);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA forward function for 2D convolution using __ldg() with 128-bit aligned accesses");
}
