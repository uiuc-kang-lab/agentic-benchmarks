#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline float compute_depthwise_conv(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int b,
    int c_out,
    int h_out,
    int w_out,
    int in_channels,
    int in_h,
    int in_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group,
    int valid_kh_start,
    int valid_kh_end) {

    int g = c_out / channels_per_group;
    int m = c_out % channels_per_group;
    float sum = 0.0f;

    #pragma unroll
    for (int kh = valid_kh_start; kh < valid_kh_end; ++kh) {
        int h_k = h_out * stride_h - padding_h + kh * dilation_h;
        #pragma unroll
        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_k = w_out * stride_w - padding_w + kw * dilation_w;
            
            if (h_k >= 0 && h_k < in_h && w_k >= 0 && w_k < in_w) {
                int input_idx = ((b * in_channels + g) * in_h + h_k) * in_w + w_k;
                int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
            }
        }
    }
    return sum;
}

__global__ void depthwise_conv2d_optimized_kernel(
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
    int channels_per_group) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_h * out_w) return;

    // Calculate output dimensions
    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int c_out = (idx / (out_w * out_h)) % out_channels;
    int b = idx / (out_w * out_h * out_channels);

    if (b >= batch_size) return;

    // Precompute valid kernel window
    int valid_kh_start = max(0, (padding_h - h_out * stride_h + dilation_h - 1) / dilation_h);
    int valid_kh_end = min(kernel_h, (in_h + padding_h - h_out * stride_h + dilation_h) / dilation_h);

    float sum = compute_depthwise_conv(
        input, weight, b, c_out, h_out, w_out,
        in_channels, in_h, in_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups, channels_per_group,
        valid_kh_start, valid_kh_end
    );

    if (bias != nullptr) {
        sum += __ldg(&bias[c_out]);
    }
    
    // Use coalesced memory write
    output[idx] = sum;
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

    int total_elements = batch_size * out_channels * out_h * out_w;
    int threads = 128;
    int blocks = (total_elements + threads - 1) / threads;

    depthwise_conv2d_optimized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h, in_w,
        out_channels,
        out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Depthwise Conv2D forward (CUDA)");
}