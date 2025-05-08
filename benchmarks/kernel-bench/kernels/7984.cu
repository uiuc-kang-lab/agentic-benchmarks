#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define TILE_SIZE 16
#define ALIGN_MASK (~0x0F)

__global__ void aligned_conv2d_kernel(
    const float4* __restrict__ input,
    const float4* __restrict__ weight,
    float4* __restrict__ output,
    const float* __restrict__ bias,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width,
    int in_width_aligned,
    int out_width_aligned) {

    __shared__ float weight_shared[TILE_SIZE][TILE_SIZE];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int channel = blockIdx.z;
    int batch = blockIdx.y;
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;
    int out_col = threadIdx.y * 4; // Process 4 elements at once with float4

    // Pre-compute aligned indices
    int in_row_base = out_row * stride - padding;
    int in_col_base = out_col * stride - padding;
    
    float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

    if (out_row < out_height && out_col < out_width_aligned/4) {
        for (int ic = 0; ic < in_channels; ++ic) {
            // Load weight tile into shared memory
            if (tid < kernel_size * kernel_size) {
                int kh = tid / kernel_size;
                int kw = tid % kernel_size;
                weight_shared[kh][kw] = __ldg(&reinterpret_cast<const float*>(weight)[
                    ((channel * in_channels + ic) * kernel_size + kh) * kernel_size + kw
                ]);
            }
            __syncthreads();

            for (int kh = 0; kh < kernel_size; kh++) {
                int in_row = in_row_base + kh * dilation;
                if (in_row >= 0 && in_row < in_height) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_col = in_col_base + kw * dilation;
                        if (in_col >= 0 && in_col < in_width) {
                            // Calculate aligned input index
                            int aligned_idx = ((batch * in_channels + ic) * in_height + in_row) * 
                                           (in_width_aligned/4) + (in_col/4);
                            float4 in_val = __ldg(&input[aligned_idx]);
                            float w_val = weight_shared[kh][kw];
                            
                            sum.x += in_val.x * w_val;
                            sum.y += in_val.y * w_val;
                            sum.z += in_val.z * w_val;
                            sum.w += in_val.w * w_val;
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Add bias if present
        if (bias != nullptr) {
            float bias_val = __ldg(&bias[channel]);
            sum.x += bias_val;
            sum.y += bias_val;
            sum.z += bias_val;
            sum.w += bias_val;
        }

        // Store aligned output
        int out_idx = ((batch * out_channels + channel) * out_height + out_row) * 
                     (out_width_aligned/4) + (out_col/4);
        output[out_idx] = sum;
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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Align widths to 128-bit boundaries
    int in_width_aligned = ((in_width + 3) / 4) * 4;
    int out_width_aligned = ((out_width + 3) / 4) * 4;

    // Create aligned input tensor
    auto x_aligned = torch::zeros({batch_size, in_channels, in_height, in_width_aligned}, x.options());
    x_aligned.slice(3, 0, in_width).copy_(x);

    // Create aligned output tensor
    auto output_aligned = torch::zeros({batch_size, out_channels, out_height, out_width_aligned}, x.options());

    dim3 block(TILE_SIZE, TILE_SIZE/4);  // Process 4 elements per thread in x dimension
    dim3 grid(
        (out_height + block.x - 1) / block.x,
        batch_size,
        out_channels
    );

    aligned_conv2d_kernel<<<grid, block>>>(
        reinterpret_cast<const float4*>(x_aligned.data_ptr<float>()),
        reinterpret_cast<const float4*>(weight.data_ptr<float>()),
        reinterpret_cast<float4*>(output_aligned.data_ptr<float>()),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        out_height,
        out_width,
        in_width_aligned,
        out_width_aligned
    );

    // Extract the actual output from aligned tensor
    return output_aligned.slice(3, 0, out_width).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Aligned memory access optimized 2D convolution");
}