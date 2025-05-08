#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void conv_transpose2d_warp_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_height,
    int kernel_width,
    int out_height,
    int out_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int bid = blockIdx.x;

    // Calculate output position
    const int out_idx = bid * BLOCK_SIZE + tid;
    const int n = out_idx / (out_channels * out_height * out_width);
    const int c = (out_idx / (out_height * out_width)) % out_channels;
    const int h = (out_idx / out_width) % out_height;
    const int w = out_idx % out_width;

    if (n < batch_size && c < out_channels && h < out_height && w < out_width) {
        float sum = 0.0f;

        // Compute input window boundaries
        const int in_h_start = (h + pad_h - kernel_height + 1 + stride_h - 1) / stride_h;
        const int in_w_start = (w + pad_w - kernel_width + 1 + stride_w - 1) / stride_w;
        const int in_h_end = min((h + pad_h) / stride_h + 1, in_height);
        const int in_w_end = min((w + pad_w) / stride_w + 1, in_width);

        // Perform convolution using warp-level parallelism
        for (int ih = in_h_start; ih < in_h_end; ih++) {
            for (int iw = in_w_start; iw < in_w_end; iw++) {
                const int kh = h + pad_h - ih * stride_h;
                const int kw = w + pad_w - iw * stride_w;

                if (kh >= 0 && kh < kernel_height && kw >= 0 && kw < kernel_width) {
                    for (int ic = lane_id; ic < in_channels; ic += WARP_SIZE) {
                        const float input_val = input[((n * in_channels + ic) * in_height + ih) * in_width + iw];
                        const float weight_val = weight[((ic * out_channels + c) * kernel_height + kh) * kernel_width + kw];
                        sum += input_val * weight_val;
                    }
                }
            }
        }

        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Write result
        if (lane_id == 0) {
            output[((n * out_channels + c) * out_height + h) * out_width + w] = sum;
        }
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto in_height = x.size(2);
    const auto in_width = x.size(3);
    const auto out_channels = weight.size(1);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);
    
    const auto out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_height;
    const auto out_width = (in_width - 1) * stride[1] - 2 * padding[1] + kernel_width;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              x.options());

    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv_transpose2d_warp_kernel<<<num_blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );

    if (!bias_obj.is_none()) {
        auto bias = bias_obj.cast<torch::Tensor>();
        output.add_(bias.view({1, out_channels, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}