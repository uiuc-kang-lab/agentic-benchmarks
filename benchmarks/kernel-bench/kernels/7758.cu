#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ inline scalar_t compute_conv3d_pixel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int b, int oc, int od, int oh, int ow,
    int batch_size, int in_channels, int in_d, int in_h, int in_w,
    int out_channels, int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int dilation,
    int groups, int in_channels_per_group) {
    
    int group = oc / (out_channels / groups);
    int in_channel_base = group * in_channels_per_group;

    // Precompute valid kernel bounds
    int in_d_base = od * stride - padding;
    int in_h_base = oh * stride - padding;
    int in_w_base = ow * stride - padding;

    scalar_t sum = 0;
    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            int id = in_d_base + kd * dilation;
            if (id < 0 || id >= in_d) continue;
            for (int kh = 0; kh < kernel_h; ++kh) {
                int ih = in_h_base + kh * dilation;
                if (ih < 0 || ih >= in_h) continue;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int iw = in_w_base + kw * dilation;
                    if (iw < 0 || iw >= in_w) continue;

                    int in_idx = (((b * in_channels + (in_channel_base + ic)) * in_d + id) * in_h + ih) * in_w + iw;
                    int w_idx = (((oc * in_channels_per_group + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                    sum += __ldg(&input[in_idx]) * __ldg(&weight[w_idx]);
                }
            }
        }
    }
    return sum;
}

template <typename scalar_t>
__global__ void conv3d_main_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int in_d, int in_h, int in_w,
    int out_channels, int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int dilation,
    int groups, int in_channels_per_group) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    
    while (idx < total_elements) {
        int residual = idx;
        const int w = residual % out_w; residual /= out_w;
        const int h = residual % out_h; residual /= out_h;
        const int d = residual % out_d; residual /= out_d;
        const int oc = residual % out_channels;
        const int b = residual / out_channels;

        output[idx] = compute_conv3d_pixel(
            input, weight, b, oc, d, h, w,
            batch_size, in_channels, in_d, in_h, in_w,
            out_channels, kernel_d, kernel_h, kernel_w,
            stride, padding, dilation,
            groups, in_channels_per_group);
        
        idx += blockDim.x * gridDim.x;
    }
}

template <typename scalar_t>
__global__ void optimized_bias_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
    int total_elements,
    int out_channels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total_elements) {
        output[idx] += __ldg(&bias[(idx / (total_elements / out_channels)) % out_channels]);
        idx += blockDim.x * gridDim.x;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {
    
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_d = input.size(2);
    const int in_h = input.size(3);
    const int in_w = input.size(4);

    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    const int out_d = (in_d + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    const int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_d, out_h, out_w}, input.options());

    const int total = batch_size * out_channels * out_d * out_h * out_w;
    const int threads = 512;
    const int blocks = (total + threads - 1) / threads;
    const int ch_per_group = in_channels / groups;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "modular_conv3d", [&] {
        conv3d_main_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, in_d, in_h, in_w,
            out_channels, out_d, out_h, out_w,
            kernel_d, kernel_h, kernel_w,
            stride, padding, dilation,
            groups, ch_per_group);

        if (bias.defined()) {
            optimized_bias_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                total,
                out_channels);
        }
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized modular 3D convolution with device functions");
}
