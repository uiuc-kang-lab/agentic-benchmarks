#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__constant__ float d_const_weight[16384];

__global__ void conv2d_kernel_aligned(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width
) {
    // Use 32x4 thread block for better occupancy and memory access patterns
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Calculate output position
    const int ow = bx * blockDim.x + tx;
    const int oh = by * blockDim.y + ty;
    
    const int n = bz / out_channels;
    const int oc = bz % out_channels;

    if(n < batch && oc < out_channels && oh < out_height && ow < out_width) {
        float sum = 0.0f;
        
        // Pre-compute input batch offset for better memory access
        const int input_batch_offset = n * in_channels * in_height * in_width;
        const int weight_oc_offset = oc * in_channels * kernel_size * kernel_size;

        #pragma unroll
        for (int ic = 0; ic < in_channels; ic++) {
            const int input_c_offset = input_batch_offset + ic * in_height * in_width;
            const int weight_ic_offset = weight_oc_offset + ic * kernel_size * kernel_size;

            #pragma unroll
            for (int kh = 0; kh < kernel_size; kh++) {
                const int ih = oh * stride - padding + kh * dilation;
                if (ih >= 0 && ih < in_height) {
                    const int input_h_offset = input_c_offset + ih * in_width;
                    const int weight_h_offset = weight_ic_offset + kh * kernel_size;

                    #pragma unroll
                    for (int kw = 0; kw < kernel_size; kw++) {
                        const int iw = ow * stride - padding + kw * dilation;
                        if (iw >= 0 && iw < in_width) {
                            // Use __ldg for read-only global memory access
                            sum += __ldg(&input[input_h_offset + iw]) *
                                   d_const_weight[weight_h_offset + kw];
                        }
                    }
                }
            }
        }

        if(bias != nullptr) {
            sum += __ldg(&bias[oc]);
        }

        // Ensure coalesced writes to global memory
        const int output_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
        output[output_idx] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if(bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    TORCH_CHECK(groups == 1, "Only groups=1 is supported");

    const int batch = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    TORCH_CHECK(weight.numel() <= 16384, "Weight tensor too large for constant memory");

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch, out_channels, out_height, out_width}, input.options());

    // Copy weights to constant memory
    cudaMemcpyToSymbol(d_const_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float));

    // Use 32x4 thread block for better memory access alignment
    dim3 block(256, 1, 1);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch * out_channels
    );

    conv2d_kernel_aligned<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        batch, in_channels, out_channels,
        in_height, in_width,
        kernel_size,
        stride, padding, dilation,
        out_height, out_width
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward convolution with aligned memory access");
}