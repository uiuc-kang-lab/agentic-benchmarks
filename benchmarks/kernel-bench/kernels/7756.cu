#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel for 3D convolution using pipelined execution with CUDA streams
template <typename scalar_t>
__global__ void conv3d_stride_bounds_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int in_d, int in_h, int in_w,
    int out_channels, int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int dilation,
    int groups, int in_channels_per_group) 
{
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_loop = blockDim.x * gridDim.x;
    
    while (idx < total_elements) {
        // Decompose linear index into 5D indices: [b, oc, od, oh, ow]
        int w_idx = idx % out_w;
        int tmp = idx / out_w;
        int h_idx = tmp % out_h;
        tmp /= out_h;
        int d_idx = tmp % out_d;
        tmp /= out_d;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        // Determine the group and the corresponding input channels
        int group = oc / (out_channels / groups);
        int in_channel_base = group * in_channels_per_group;

        // Compute the top-left-front corner of the receptive field in the input
        int in_d_base = d_idx * stride - padding;
        int in_h_base = h_idx * stride - padding;
        int in_w_base = w_idx * stride - padding;

        // Precompute valid kernel loop bounds for depth
        int kd_start = 0;
        if (in_d_base < 0) {
            kd_start = (-in_d_base + dilation - 1) / dilation;
        }
        int kd_end = kernel_d;
        if (in_d_base + (kernel_d - 1) * dilation >= in_d) {
            kd_end = (in_d - in_d_base + dilation - 1) / dilation;
            if(kd_end > kernel_d) kd_end = kernel_d;
        }

        // Precompute valid kernel loop bounds for height
        int kh_start = 0;
        if (in_h_base < 0) {
            kh_start = (-in_h_base + dilation - 1) / dilation;
        }
        int kh_end = kernel_h;
        if (in_h_base + (kernel_h - 1) * dilation >= in_h) {
            kh_end = (in_h - in_h_base + dilation - 1) / dilation;
            if(kh_end > kernel_h) kh_end = kernel_h;
        }

        // Precompute valid kernel loop bounds for width
        int kw_start = 0;
        if (in_w_base < 0) {
            kw_start = (-in_w_base + dilation - 1) / dilation;
        }
        int kw_end = kernel_w;
        if (in_w_base + (kernel_w - 1) * dilation >= in_w) {
            kw_end = (in_w - in_w_base + dilation - 1) / dilation;
            if(kw_end > kernel_w) kw_end = kernel_w;
        }

        // Perform convolution sum
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
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
        idx += stride_loop;
    }
}

// Kernel for bias addition using a stride loop
template <typename scalar_t>
__global__ void add_bias_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
    int total_elements,
    int out_channels,
    int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_loop = blockDim.x * gridDim.x;
    
    while(idx < total_elements) {
        // Remove the width dimension to extract the channel index
        int tmp = idx / out_w;
        int oc = tmp % out_channels;
        output[idx] += bias[oc];
        idx += stride_loop;
    }
}

// Host forward function setting up convolution parameters and launching kernels
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

    // Input dimensions: [batch, in_channels, in_d, in_h, in_w]
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    // Weight dimensions: [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w]
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Calculate output dimensions using standard convolution formula
    int out_d = (in_d + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto options = input.options();
    auto output = at::empty({batch_size, out_channels, out_d, out_h, out_w}, options);

    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    int in_channels_per_group = in_channels / groups;

    // Create a CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pipelined_conv3d_cuda", ([&] {
        const auto* input_ptr = input.data_ptr<scalar_t>();
        const auto* weight_ptr = weight.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();

        conv3d_stride_bounds_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input_ptr, weight_ptr, output_ptr,
            batch_size, in_channels, in_d, in_h, in_w,
            out_channels, out_d, out_h, out_w,
            kernel_d, kernel_h, kernel_w,
            stride, padding, dilation,
            groups, in_channels_per_group);

        if (bias.defined()) {
            const auto* bias_ptr = bias.data_ptr<scalar_t>();
            add_bias_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                output_ptr, bias_ptr, total_elements, out_channels, out_w);
        }
    }));

    // Synchronize the stream
    cudaStreamSynchronize(stream);
    // Destroy the stream
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward CUDA kernel with pipelined streams");
}