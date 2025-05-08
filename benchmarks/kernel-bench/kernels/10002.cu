#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements_per_thread = 4;
    const int total_elements = batch_size * out_channels * out_h * out_w;
    
    for (int i = tid * elements_per_thread; i < total_elements; i += blockDim.x * gridDim.x * elements_per_thread) {
        int remaining = i;
        const int w_out_base = remaining % out_w;
        remaining /= out_w;
        const int h_out = remaining % out_h;
        remaining /= out_h;
        const int c_out = remaining % out_channels;
        const int b = remaining / out_channels;

        if (b >= batch_size) continue;

        const int g = c_out / channels_per_group;
        const int m = c_out % channels_per_group;

        const int input_batch_offset = b * in_channels * in_h * in_w;
        const int input_channel_offset = g * in_h * in_w;
        const int weight_offset = (g * channels_per_group + m) * kernel_h * kernel_w;

        #pragma unroll
        for (int elem = 0; elem < elements_per_thread && w_out_base + elem < out_w; ++elem) {
            const int w_out = w_out_base + elem;
            float sum = 0.0f;

            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh) {
                const int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                if (h_in >= 0 && h_in < in_h) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        const int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                        if (w_in >= 0 && w_in < in_w) {
                            const int input_idx = input_batch_offset + input_channel_offset + h_in * in_w + w_in;
                            const int weight_idx = weight_offset + kh * kernel_w + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }

            if (bias != nullptr) {
                sum += bias[c_out];
            }

            const int out_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
            output[out_idx] = sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    const int elements_per_thread = 4;
    const int total_elements = batch_size * out_channels * out_h * out_w;
    const int threads = 256;
    const int blocks = (total_elements + (threads * elements_per_thread) - 1) / (threads * elements_per_thread);

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise Conv2D forward (CUDA)");
}