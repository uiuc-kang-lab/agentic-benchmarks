#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handles.h>
#include <ATen/cudnn/Descriptors.h>
#include <cudnn.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 4

// Device function for calculating output index
__device__ int calculate_output_index(int idx, int out_width, int out_height, int out_depth, int out_channels, int &w_out, int &h_out, int &d_out, int &c_out, int &b) {
    w_out = idx % out_width;
    int tmp = idx / out_width;
    h_out = tmp % out_height;
    tmp = tmp / out_height;
    d_out = tmp % out_depth;
    tmp = tmp / out_depth;
    c_out = tmp % out_channels;
    b = tmp / out_channels;
    return 0;
}

// Device function for computing convolution
__device__ float compute_convolution(const float* input, const float* weight, int b, int in_channels_per_group, int in_depth, int in_height, int in_width, int kernel_d, int kernel_h, int kernel_w, int d_out, int h_out, int w_out, int stride, int padding, int dilation, int group, int c_out) {
    float sum = 0.0f;
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        int in_channel = group * in_channels_per_group + ic;
        for (int kd = 0; kd < kernel_d; kd++) {
            int d_in = d_out * stride - padding + kd * dilation;
            if (d_in < 0 || d_in >= in_depth) continue;
            for (int kh = 0; kh < kernel_h; kh++) {
                int h_in = h_out * stride - padding + kh * dilation;
                if (h_in < 0 || h_in >= in_height) continue;
                for (int kw = 0; kw < kernel_w; kw++) {
                    int w_in = w_out * stride - padding + kw * dilation;
                    if (w_in < 0 || w_in >= in_width) continue;
                    int input_index = ((b * in_channels_per_group + in_channel) * in_depth + d_in) * in_height * in_width + h_in * in_width + w_in;
                    int weight_index = (((c_out * in_channels_per_group) + ic) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                    sum += input[input_index] * weight[weight_index];
                }
            }
        }
    }
    return sum;
}

// Kernel function for 3D convolution
__global__ void conv3d_modular_kernel(float* output, const float* input, const float* weight, const float* bias, int batch_size, int in_channels, int out_channels, int in_depth, int in_height, int in_width, int kernel_d, int kernel_h, int kernel_w, int out_depth, int out_height, int out_width, int stride, int padding, int dilation, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int stride_size = gridDim.x * blockDim.x;
    
    for (; idx < total_elements; idx += stride_size) {
        int w_out, h_out, d_out, c_out, b;
        calculate_output_index(idx, out_width, out_height, out_depth, out_channels, w_out, h_out, d_out, c_out, b);
        
        int group = c_out / (out_channels / groups);
        int in_channels_per_group = in_channels / groups;
        float sum = compute_convolution(input, weight, b, in_channels_per_group, in_depth, in_height, in_width, kernel_d, kernel_h, kernel_w, d_out, h_out, w_out, stride, padding, dilation, group, c_out);
        
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        output[idx] = sum;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int num_blocks = (total_elements + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    
    conv3d_modular_kernel<<<num_blocks, BLOCK_SIZE_X>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation, groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with modular device functions (CUDA)");
}
