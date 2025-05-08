#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void conv3d_coalesced_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int in_d, int in_h, int in_w,
    int out_channels, int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int dilation,
    int groups, int in_channels_per_group) 
{
    const int vector_size = 4;
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * vector_size;
    
    if (idx >= total_elements) return;

    // Vectorized output index calculation
    int out_idx[vector_size];
    #pragma unroll
    for (int v = 0; v < vector_size; ++v) {
        out_idx[v] = min(idx + v, total_elements - 1);
    }

    // Process vector_size elements per thread
    #pragma unroll
    for (int v = 0; v < vector_size; ++v) {
        int linear_idx = out_idx[v];
        
        // Decompose linear index into 5D indices
        int w = linear_idx % out_w;
        int tmp = linear_idx / out_w;
        int h = tmp % out_h;
        tmp /= out_h;
        int d = tmp % out_d;
        tmp /= out_d;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        // Compute kernel bounds
        int group = oc / (out_channels / groups);
        int in_base_d = d * stride - padding;
        int in_base_h = h * stride - padding;
        int in_base_w = w * stride - padding;

        // Precompute valid kernel ranges
        int kd_start = max(0, (-in_base_d + dilation - 1) / dilation);
        int kd_end = min(kernel_d, (in_d - in_base_d + dilation - 1) / dilation);
        int kh_start = max(0, (-in_base_h + dilation - 1) / dilation);
        int kh_end = min(kernel_h, (in_h - in_base_h + dilation - 1) / dilation);
        int kw_start = max(0, (-in_base_w + dilation - 1) / dilation);
        int kw_end = min(kernel_w, (in_w - in_base_w + dilation - 1) / dilation);

        scalar_t sum = 0;
        int in_channel_base = group * in_channels_per_group;

        // Optimized inner loops with coalesced memory access
        for (int ic = 0; ic < in_channels_per_group; ++ic) {
            int in_channel = in_channel_base + ic;
            
            for (int kd = kd_start; kd < kd_end; ++kd) {
                int id = in_base_d + kd * dilation;
                for (int kh = kh_start; kh < kh_end; ++kh) {
                    int ih = in_base_h + kh * dilation;
                    
                    // Vectorized width processing
                    for (int kw = kw_start; kw < kw_end; ++kw) {
                        int iw = in_base_w + kw * dilation;
                        
                        // Coalesced input access pattern
                        int input_idx = ((b * in_channels + in_channel) * in_d + id) * in_h * in_w + ih * in_w + iw;
                        int weight_idx = (((oc * in_channels_per_group + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                        
                        sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                    }
                }
            }
        }
        output[linear_idx] = sum;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) 
{
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");

    // Dimension calculations
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    auto in_spatial = input.sizes().slice(2);
    auto kernel_size = weight.sizes().slice(2);

    int out_channels = weight.size(0);
    int in_channels_per_group = in_channels / groups;

    // Calculate output dimensions
    auto output_sizes = input.sizes().vec();
    for (int i = 2; i < 5; ++i) {
        output_sizes[i] = (input.sizes()[i] + 2 * padding - dilation * (kernel_size[i-2] - 1) - 1) / stride + 1;
    }
    output_sizes[1] = out_channels;

    auto output = at::empty(output_sizes, input.options());

    // Kernel configuration
    int total_elements = output.numel();
    const int vector_size = 4;
    int threads = 256;
    int blocks = (total_elements + threads * vector_size - 1) / (threads * vector_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_coalesced", ([&] {
        conv3d_coalesced_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, in_spatial[0], in_spatial[1], in_spatial[2],
            out_channels, output_sizes[2], output_sizes[3], output_sizes[4],
            kernel_size[0], kernel_size[1], kernel_size[2],
            stride, padding, dilation, groups, in_channels_per_group);
    }));

    if (bias.defined()) {
        output += bias.view({1, out_channels, 1, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced 3D convolution with vectorized memory access");
}