#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int out_depth,
    int out_height,
    int out_width,
    int k_depth,
    int k_height,
    int k_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int groups,
    int group_channels) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_depth * out_height * out_width;
    
    for (int linear_idx = idx; linear_idx < total; linear_idx += gridDim.x * blockDim.x) {
        int n = linear_idx / (out_channels * out_depth * out_height * out_width);
        int c = (linear_idx / (out_depth * out_height * out_width)) % out_channels;
        int d = (linear_idx / (out_height * out_width)) % out_depth;
        int h = (linear_idx / out_width) % out_height;
        int w = linear_idx % out_width;

        int g = c / group_channels;
        float value = 0.0f;

        for (int kd = 0; kd < k_depth; ++kd) {
            for (int kh = 0; kh < k_height; ++kh) {
                for (int kw = 0; kw < k_width; ++kw) {
                    int in_d = (d + pad_d - kd) / stride_d;
                    int in_h = (h + pad_h - kh) / stride_h;
                    int in_w = (w + pad_w - kw) / stride_w;

                    if ((d + pad_d - kd) % stride_d == 0 &&
                        (h + pad_h - kh) % stride_h == 0 &&
                        (w + pad_w - kw) % stride_w == 0) {
                        
                        if (in_d >= 0 && in_d < in_depth &&
                            in_h >= 0 && in_h < in_height &&
                            in_w >= 0 && in_w < in_width) {
                            
                            for (int ic = 0; ic < group_channels; ++ic) {
                                int in_c = g * group_channels + ic;
                                int input_idx = ((n * in_channels + in_c) * in_depth + in_d) * in_height * in_width + in_h * in_width + in_w;
                                int weight_idx = ((c * k_depth + kd) * k_height + kh) * k_width + kw + ic * k_depth * k_height * k_width;
                                value += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }

        if (bias) {
            value += bias[c];
        }

        output[linear_idx] = value;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) { CHECK_INPUT(bias.value()); }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_depth = x.size(2);
    int in_height = x.size(3);
    int in_width = x.size(4);

    int out_channels = weight.size(1) * groups;
    int k_depth = weight.size(2);
    int k_height = weight.size(3);
    int k_width = weight.size(4);
    
    int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + k_depth + output_padding[0];
    int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + k_height + output_padding[1];
    int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + k_width + output_padding[2];

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, x.options());
    
    int threads = 256;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int blocks = (total_elements + threads - 1) / threads;

    conv_transpose3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        out_depth,
        out_height,
        out_width,
        k_depth,
        k_height,
        k_width,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        groups,
        out_channels / groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Transposed Conv3D forward (CUDA)");
}
