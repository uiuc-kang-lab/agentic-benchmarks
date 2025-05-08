#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace py = pybind11;

__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int out_h,
    const int out_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const bool has_bias,
    const int total_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Pre-compute strides for better memory access patterns
    const int input_channel_stride = in_h * in_w;
    const int weight_kernel_stride = kernel_h * kernel_w;
    const int output_channel_stride = out_h * out_w;
    
    // Each thread processes multiple elements with striding
    for (int idx = tid; idx < total_elements; idx += stride) {
        // Decode output position
        const int out_x = idx % out_w;
        int tmp = idx / out_w;
        const int out_y = tmp % out_h;
        tmp = tmp / out_h;
        const int oc = tmp % out_channels;
        const int n = tmp / out_channels;

        // Initialize accumulator
        float sum = has_bias ? bias[oc] : 0.0f;
        
        // Base positions for input mapping
        const int base_y = out_y + pad_h;
        const int base_x = out_x + pad_w;

        // Compute input contribution for this output position
        #pragma unroll 4
        for (int ky = 0; ky < kernel_h; ky++) {
            const int t_y = base_y - ky;
            if (t_y % stride_h == 0) {
                const int in_y = t_y / stride_h;
                if (in_y >= 0 && in_y < in_h) {
                    #pragma unroll 4
                    for (int kx = 0; kx < kernel_w; kx++) {
                        const int t_x = base_x - kx;
                        if (t_x % stride_w == 0) {
                            const int in_x = t_x / stride_w;
                            if (in_x >= 0 && in_x < in_w) {
                                // Pre-compute offsets for efficient memory access
                                const int input_base = (n * in_channels * input_channel_stride) + 
                                                     (in_y * in_w + in_x);
                                const int weight_base = (oc * weight_kernel_stride) + 
                                                      (ky * kernel_w + kx);
                                
                                // Accumulate contributions from all input channels
                                #pragma unroll 4
                                for (int ic = 0; ic < in_channels; ic++) {
                                    const int input_idx = input_base + ic * input_channel_stride;
                                    const int weight_idx = (ic * out_channels * weight_kernel_stride) + 
                                                         weight_base;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Write result
        output[idx] = sum;
    }
}

torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    const int N = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_channels = weight.size(1);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];

    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w;

    auto output = torch::zeros({N, out_channels, out_h, out_w}, input.options());

    const int total_elements = N * out_channels * out_h * out_w;
    const int thread_count = 256; // Ensure thread count is a multiple of 32 for warp alignment
    const int block_count = std::min(65535, (total_elements + thread_count - 1) / thread_count);

    const bool has_bias = (bias.has_value() && bias.value().numel() > 0);
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;

    conv_transpose2d_forward_kernel<<<block_count, thread_count, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, in_channels, in_h, in_w,
        out_channels, kernel_h, kernel_w,
        out_h, out_w, stride_h, stride_w,
        pad_h, pad_w, has_bias,
        total_elements
    );

    return output;
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    c10::optional<torch::Tensor> bias = c10::nullopt;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
    }
    return conv_transpose2d_forward_cuda(input, weight, bias, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward (optimized stride)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}