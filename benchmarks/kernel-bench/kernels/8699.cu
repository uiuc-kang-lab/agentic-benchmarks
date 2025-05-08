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

    int g = blockIdx.z * blockDim.z + threadIdx.z;
    if (g >= groups) return;

    int c_out_start = g * group_channels;
    int c_in_start = g * (in_channels / groups);

    for (int n = blockIdx.x; n < batch_size; n += gridDim.x) {
        for (int c_out = c_out_start; c_out < c_out_start + group_channels; c_out++) {
            for (int d = threadIdx.y; d < out_depth; d += blockDim.y) {
                for (int h = threadIdx.x; h < out_height; h += blockDim.x) {
                    for (int w = blockIdx.y; w < out_width; w += gridDim.y) {
                        float val = 0.0f;

                        for (int kd = 0; kd < k_depth; kd++) {
                            for (int kh = 0; kh < k_height; kh++) {
                                for (int kw = 0; kw < k_width; kw++) {
                                    int in_d = (d + pad_d - kd) / stride_d;
                                    int in_h = (h + pad_h - kh) / stride_h;
                                    int in_w = (w + pad_w - kw) / stride_w;

                                    if ((d + pad_d - kd) % stride_d == 0 &&
                                        (h + pad_h - kh) % stride_h == 0 &&
                                        (w + pad_w - kw) % stride_w == 0) {
                                        
                                        in_d = (d + pad_d - kd) / stride_d;
                                        in_h = (h + pad_h - kh) / stride_h;
                                        in_w = (w + pad_w - kw) / stride_w;

                                        if (in_d >= 0 && in_d < in_depth &&
                                            in_h >= 0 && in_h < in_height &&
                                            in_w >= 0 && in_w < in_width) {
                                            
                                            int c_in = c_in_start + (c_out - c_out_start);
                                            int input_idx = ((n * (in_channels / groups) + (c_in - c_in_start)) * in_depth + in_d) * in_height * in_width + in_h * in_width + in_w;
                                            int weight_idx = ((c_out * k_depth + kd) * k_height + kh) * k_width + kw;
                                            val += input[input_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                        }

                        if (bias) {
                            val += bias[c_out];
                        }

                        int output_idx = ((n * out_channels + c_out) * out_depth + d) * out_height * out_width + h * out_width + w;
                        atomicAdd(&output[output_idx], val);
                    }
                }
            }
        }
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
    if (bias.has_value()) { auto bias_tensor = *bias; CHECK_INPUT(bias_tensor); }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_depth = x.size(2);
    int in_height = x.size(3);
    int in_width = x.size(4);

    int out_channels = weight.size(1) * groups;
    int k_depth = weight.size(2);
    int k_height = weight.size(3);
    int k_width = weight.size(4);

    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];

    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];

    int out_depth = (in_depth - 1) * stride_d - 2 * pad_d + k_depth + output_padding[0];
    int out_height = (in_height - 1) * stride_h - 2 * pad_h + k_height + output_padding[1];
    int out_width = (in_width - 1) * stride_w - 2 * pad_w + k_width + output_padding[2];

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, x.options());

    dim3 block(8, 8, 4);
    dim3 grid(
        (batch_size + block.x - 1) / block.x,
        (out_width + block.y - 1) / block.y,
        (groups + block.z - 1) / block.z
    );

    conv_transpose3d_kernel<<<grid, block>>>(
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
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        groups,
        out_channels / groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Transposed Conv3D forward (CUDA)");
}
