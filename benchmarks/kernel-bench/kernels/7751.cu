#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses explicit __ldg() intrinsics to enforce read-only caching and improve memory coalescing.
// It assumes that the output tensor is stored in contiguous order so that consecutive threads write to consecutive addresses.

template <typename scalar_t>
__global__ void coalesced_conv3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int in_d, int in_h, int in_w,
    int out_channels, int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int dilation,
    int groups, int in_channels_per_group) 
{
    int total = batch_size * out_channels * out_d * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_loop = blockDim.x * gridDim.x;

    while (idx < total) {
        // Decompose linear index into output indices [b, oc, d, h, w]
        int w = idx % out_w;
        int tmp = idx / out_w;
        int h = tmp % out_h;
        tmp /= out_h;
        int d = tmp % out_d;
        tmp /= out_d;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        // Determine group and corresponding input channel block
        int group = oc / (out_channels / groups);
        int in_channel_base = group * in_channels_per_group;

        // Compute top-left-front corner of the receptive field in input
        int in_d_base = d * stride - padding;
        int in_h_base = h * stride - padding;
        int in_w_base = w * stride - padding;

        // Precompute valid kernel bounds for depth
        int kd_start = 0;
        if (in_d_base < 0) {
            kd_start = (-in_d_base + dilation - 1) / dilation;
        }
        int kd_end = kernel_d;
        if (in_d_base + (kernel_d - 1) * dilation >= in_d) {
            kd_end = (in_d - in_d_base + dilation - 1) / dilation;
            if (kd_end > kernel_d) kd_end = kernel_d;
        }

        // Precompute valid kernel bounds for height
        int kh_start = 0;
        if (in_h_base < 0) {
            kh_start = (-in_h_base + dilation - 1) / dilation;
        }
        int kh_end = kernel_h;
        if (in_h_base + (kernel_h - 1) * dilation >= in_h) {
            kh_end = (in_h - in_h_base + dilation - 1) / dilation;
            if (kh_end > kernel_h) kh_end = kernel_h;
        }

        // Precompute valid kernel bounds for width
        int kw_start = 0;
        if (in_w_base < 0) {
            kw_start = (-in_w_base + dilation - 1) / dilation;
        }
        int kw_end = kernel_w;
        if (in_w_base + (kernel_w - 1) * dilation >= in_w) {
            kw_end = (in_w - in_w_base + dilation - 1) / dilation;
            if (kw_end > kernel_w) kw_end = kernel_w;
        }

        // Compute convolution sum using __ldg() to coalesce global memory reads
        scalar_t sum = 0;
        for (int ic = 0; ic < in_channels_per_group; ++ic) {
            int in_channel = in_channel_base + ic;
            for (int kd = kd_start; kd < kd_end; ++kd) {
                int id = in_d_base + kd * dilation;
                for (int kh = kh_start; kh < kh_end; ++kh) {
                    int ih = in_h_base + kh * dilation;
                    for (int kw = kw_start; kw < kw_end; ++kw) {
                        int iw = in_w_base + kw * dilation;
                        int input_idx = (((b * in_channels + in_channel) * in_d + id) * in_h + ih) * in_w + iw;
                        int weight_idx = (((oc * in_channels_per_group + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                        sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                    }
                }
            }
        }
        output[idx] = sum;
        idx += stride_loop;
    }
}

// Kernel for bias addition that uses __ldg() for coalesced bias reads
template <typename scalar_t>
__global__ void coalesced_add_bias_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
    int total,
    int out_channels,
    int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_loop = blockDim.x * gridDim.x;
    while (idx < total) {
        int tmp = idx / out_w;
        int oc = tmp % out_channels;
        output[idx] += __ldg(&bias[oc]);
        idx += stride_loop;
    }
}

// Host forward function that sets up parameters and launches the kernels
at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int out_d = (in_d + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto options = input.options();
    auto output = at::empty({batch_size, out_channels, out_d, out_h, out_w}, options);

    int total = batch_size * out_channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    int in_channels_per_group = in_channels / groups;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "coalesced_conv3d_cuda", ([&] {
        const auto* input_ptr = input.data_ptr<scalar_t>();
        const auto* weight_ptr = weight.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();

        coalesced_conv3d_kernel<scalar_t><<<blocks, threads>>>(
            input_ptr, weight_ptr, output_ptr,
            batch_size, in_channels, in_d, in_h, in_w,
            out_channels, out_d, out_h, out_w,
            kernel_d, kernel_h, kernel_w,
            stride, padding, dilation,
            groups, in_channels_per_group);
        cudaDeviceSynchronize();

        if (bias.defined()) {
            const auto* bias_ptr = bias.data_ptr<scalar_t>();
            coalesced_add_bias_kernel<scalar_t><<<blocks, threads>>>(
                output_ptr, bias_ptr, total, out_channels, out_w);
            cudaDeviceSynchronize();
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward CUDA kernel with coalesced memory accesses using __ldg");
}
