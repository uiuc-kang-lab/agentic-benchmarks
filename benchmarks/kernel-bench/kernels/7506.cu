#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline int d_min(int a, int b) { return a < b ? a : b; }
__device__ inline int d_max(int a, int b) { return a > b ? a : b; }

__global__ void convTranspose2dKernelCoalesce(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int height_in,
    int width_in,
    int kernel_size,
    int stride,
    int padding,
    int height_out,
    int width_out,
    int groups,
    bool bias_present
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * height_out * width_out;
    if (idx >= total) return;

    int w = idx % width_out;
    int tmp = idx / width_out;
    int h = tmp % height_out;
    tmp /= height_out;
    int out_ch = tmp % out_channels;
    int b = tmp / out_channels;

    float out_val = 0.0f;

    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = out_ch / out_channels_per_group;
    int out_ch_mod = out_ch % out_channels_per_group;

    int h_temp = h + padding;
    int w_temp = w + padding;

    int p0 = h_temp % stride;
    int q0 = w_temp % stride;

    int p_min = d_max(p0, h_temp - (height_in - 1) * stride);
    int p_max = d_min(kernel_size - 1, h_temp);
    int p_start = p_min;
    int mod = p_start % stride;
    if(mod != p0) {
        p_start += ((stride + p0 - mod) % stride);
    }

    int q_min = d_max(q0, w_temp - (width_in - 1) * stride);
    int q_max = d_min(kernel_size - 1, w_temp);
    int q_start = q_min;
    int modq = q_start % stride;
    if(modq != q0) {
        q_start += ((stride + q0 - modq) % stride);
    }

    int in_ch_start = group * in_channels_per_group;
    int in_ch_end = in_ch_start + in_channels_per_group;

    for (int in_ch = in_ch_start; in_ch < in_ch_end; in_ch++) {
        int in_ch_offset = (b * in_channels + in_ch) * height_in;
        int weight_start = (in_ch * out_channels_per_group + out_ch_mod) * kernel_size * kernel_size;

        for (int p = p_start; p <= p_max; p += stride) {
            int i_in = (h_temp - p) / stride;
            int input_base = in_ch_offset + i_in * width_in;
            int weight_p_offset = weight_start + p * kernel_size;

            for (int q = q_start; q <= q_max; q += stride) {
                int j_in = (w_temp - q) / stride;
                int input_idx = input_base + j_in;
                int weight_idx = weight_p_offset + q;
                out_val += input[input_idx] * weight[weight_idx];
            }
        }
    }
    if (bias_present) {
        out_val += bias[out_ch];
    }
    int output_idx = ((b * out_channels + out_ch) * height_out + h) * width_out + w;
    output[output_idx] = out_val;
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    bool bias_present = false;
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias_tensor.is_contiguous(), "Bias tensor must be contiguous");
        bias_present = true;
    }

    int batch = x.size(0);
    int in_channels = x.size(1);
    int height_in = x.size(2);
    int width_in = x.size(3);

    int kernel_size = weight.size(2);
    int out_channels = weight.size(1) * groups;

    int height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int width_out  = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor output = torch::empty({batch, out_channels, height_out, width_out}, options);

    int total_threads = batch * out_channels * height_out * width_out;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    convTranspose2dKernelCoalesce<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_present ? bias_tensor.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        height_in,
        width_in,
        kernel_size,
        stride,
        padding,
        height_out,
        width_out,
        groups,
        bias_present
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward with memory coalescing (CUDA)");
}
