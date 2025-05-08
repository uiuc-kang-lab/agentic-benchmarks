#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline int d_min(int a, int b) { return a < b ? a : b; }
__device__ inline int d_max(int a, int b) { return a > b ? a : b; }

__device__ void compute_valid_pq_ranges(
    int h_temp, int w_temp, int stride, int kernel_size,
    int height_in, int width_in,
    int& p_start, int& p_step, int& p_end,
    int& q_start, int& q_step, int& q_end
) {
    // Calculate valid p range
    int p0 = h_temp % stride;
    p_start = d_max(p0, h_temp - (height_in - 1) * stride);
    p_end = d_min(kernel_size - 1, h_temp);
    int mod_p = p_start % stride;
    p_start += (mod_p != p0) ? ((stride + p0 - mod_p) % stride) : 0;
    p_step = stride;

    // Calculate valid q range
    int q0 = w_temp % stride;
    q_start = d_max(q0, w_temp - (width_in - 1) * stride);
    q_end = d_min(kernel_size - 1, w_temp);
    int mod_q = q_start % stride;
    q_start += (mod_q != q0) ? ((stride + q0 - mod_q) % stride) : 0;
    q_step = stride;
}

__device__ float compute_channel_contribution(
    const float* input, const float* weight,
    int b, int in_ch, int out_ch_mod,
    int h_temp, int w_temp, int stride,
    int height_in, int width_in,
    int kernel_size, int out_channels_per_group,
    int p_start, int p_step, int p_end,
    int q_start, int q_step, int q_end
) {
    float sum = 0.0f;
    int b_in_offset = b * height_in * width_in;
    
    for (int p = p_start; p <= p_end; p += p_step) {
        int i_in = (h_temp - p) / stride;
        int row_offset = b_in_offset + i_in * width_in;
        int weight_p_offset = (out_ch_mod * kernel_size + p) * kernel_size;

        for (int q = q_start; q <= q_end; q += q_step) {
            int j_in = (w_temp - q) / stride;
            int input_idx = row_offset + j_in;
            int weight_idx = weight_p_offset + q;
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    return sum;
}

__global__ void convTranspose2dOptimizedKernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int height_in, int width_in, int kernel_size,
    int stride, int padding, int height_out, int width_out,
    int groups, bool bias_present
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_channels * height_out * width_out) return;

    // Position calculations
    int w = idx % width_out;
    int h = (idx / width_out) % height_out;
    int out_ch = (idx / (width_out * height_out)) % out_channels;
    int b = idx / (width_out * height_out * out_channels);

    // Group parameters
    int group = out_ch / (out_channels / groups);
    int in_ch_start = group * (in_channels / groups);
    int in_ch_end = in_ch_start + (in_channels / groups);

    // Precompute position offsets
    int h_temp = h + padding;
    int w_temp = w + padding;

    // Compute valid pq ranges
    int p_start, p_step, p_end;
    int q_start, q_step, q_end;
    compute_valid_pq_ranges(h_temp, w_temp, stride, kernel_size,
                           height_in, width_in,
                           p_start, p_step, p_end,
                           q_start, q_step, q_end);

    // Main computation loop
    float out_val = 0.0f;
    for (int in_ch = in_ch_start; in_ch < in_ch_end; ++in_ch) {
        out_val += compute_channel_contribution(
            input + in_ch * height_in * width_in * batch,
            weight + in_ch * (out_channels/groups) * kernel_size * kernel_size,
            b, in_ch, out_ch % (out_channels/groups),
            h_temp, w_temp, stride,
            height_in, width_in, kernel_size, out_channels/groups,
            p_start, p_step, p_end, q_start, q_step, q_end
        );
    }

    if (bias_present) out_val += bias[out_ch];
    output[idx] = out_val;
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x, torch::Tensor weight, torch::optional<torch::Tensor> bias,
    int64_t stride, int64_t padding, int64_t output_padding, int64_t groups
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "Inputs must be contiguous");

    // Dimension calculations
    int batch = x.size(0);
    int in_channels = x.size(1);
    int height_in = x.size(2);
    int width_in = x.size(3);
    int kernel_size = weight.size(2);
    int out_channels = weight.size(1) * groups;

    int height_out = (height_in-1)*stride - 2*padding + kernel_size + output_padding;
    int width_out = (width_in-1)*stride - 2*padding + kernel_size + output_padding;

    auto output = torch::zeros({batch, out_channels, height_out, width_out}, x.options());

    // Kernel launch parameters
    int total_elements = batch * out_channels * height_out * width_out;
    int block_size = 128;  // Optimized for H100's 1024 threads per SM
    int grid_size = (total_elements + block_size - 1) / block_size;

    convTranspose2dOptimizedKernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        height_in, width_in, kernel_size,
        stride, padding, height_out, width_out,
        groups, bias.has_value()
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Optimized ConvTranspose2D forward");
}
