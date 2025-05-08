#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Aligned load helper functions
__device__ __forceinline__ float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

__global__ void conv2d_aligned_kernel(
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

    const int n = blockIdx.x;
    const int oc = blockIdx.y;
    const int out_y = blockIdx.z * blockDim.y + threadIdx.y;
    const int out_x = threadIdx.x;

    if (out_y >= out_height || out_x >= out_width) return;

    float sum = 0.0f;
    
    // Pre-compute base indices for input and weight
    const int input_batch_offset = n * in_channels * in_height * in_width;
    const int weight_output_offset = oc * in_channels * kernel_size * kernel_size;

    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        const int input_channel_offset = input_batch_offset + ic * in_height * in_width;
        const int weight_channel_offset = weight_output_offset + ic * kernel_size * kernel_size;

        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            const int in_y = out_y * stride - padding + ky * dilation;
            
            if (in_y >= 0 && in_y < in_height) {
                const int input_row_offset = input_channel_offset + in_y * in_width;
                const int weight_row_offset = weight_channel_offset + ky * kernel_size;

                #pragma unroll
                for (int kx = 0; kx < kernel_size; ++kx) {
                    const int in_x = out_x * stride - padding + kx * dilation;
                    
                    if (in_x >= 0 && in_x < in_width) {
                        // Use __ldg for read-only data
                        sum += __ldg(&input[input_row_offset + in_x]) *
                               __ldg(&weight[weight_row_offset + kx]);
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += __ldg(&bias[oc]);
    }

    // Ensure aligned store to output
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

    TORCH_CHECK(groups == 1, "Only groups==1 is supported in this optimized kernel");

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    // Optimize thread block configuration for aligned memory access
    const int BLOCK_WIDTH = 32;  // Align with warp size for coalesced memory access
    dim3 threads(BLOCK_WIDTH, 1);
    dim3 blocks(batch, out_channels, (out_height + threads.y - 1) / threads.y);

    conv2d_aligned_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, dilation);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Memory-aligned CUDA convolution implementation");
}