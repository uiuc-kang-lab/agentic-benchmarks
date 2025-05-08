#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_size = gridDim.x * blockDim.x;
    int total_elements = batch_size * out_channels * output_h * output_w;

    for (int idx = tid; idx < total_elements; idx += stride_size) {
        int w_out = idx % output_w;
        int temp = idx / output_w;
        int h_out = temp % output_h;
        temp /= output_h;
        int oc = temp % out_channels;
        int b = temp / out_channels;

        int in_ch = oc / channels_per_group;
        int weight_ch = oc % channels_per_group;

        float sum = 0.0f;
        int h_in_start = h_out * stride - padding;
        int w_in_start = w_out * stride - padding;

        // Calculate effective kernel bounds to avoid divergent branches
        int kh_start = max(0, -h_in_start);
        int kh_end = min(kernel_size, input_h - h_in_start);
        int kw_start = max(0, -w_in_start);
        int kw_end = min(kernel_size, input_w - w_in_start);

        #pragma unroll
        for (int kh = kh_start; kh < kh_end; ++kh) {
            int h_in = h_in_start + kh;
            #pragma unroll
            for (int kw = kw_start; kw < kw_end; ++kw) {
                int w_in = w_in_start + kw;
                
                int input_idx = b * (in_channels * input_h * input_w)
                              + in_ch * (input_h * input_w)
                              + h_in * input_w
                              + w_in;
                              
                int weight_idx = in_ch * (channels_per_group * kernel_size * kernel_size)
                               + weight_ch * (kernel_size * kernel_size)
                               + kh * kernel_size
                               + kw;

                sum += input[input_idx] * weight[weight_idx];
            }
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        output[idx] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    int threads = 256;
    int max_blocks = 65535;
    int total_elements = batch_size * out_channels * output_h * output_w;
    int blocks = min((total_elements + threads - 1) / threads, max_blocks);

    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        output_h,
        output_w,
        kernel_size,
        stride,
        padding,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}