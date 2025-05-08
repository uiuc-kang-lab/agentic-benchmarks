#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__device__ inline int d_min(int a, int b) { return a < b ? a : b; }
__device__ inline int d_max(int a, int b) { return a > b ? a : b; }

__global__ void convTranspose2dKernel2DTiling(
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
    bool bias_present) {

    // 2D block for spatial dimensions, grid z for batches
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (w >= width_out || h >= height_out || b >= batch) return;

    // Precompute values common to all output channels for this output pixel
    int h_temp = h + padding;
    int w_temp = w + padding;
    int p0 = h_temp % stride;
    int q0 = w_temp % stride;
    int p_min = d_max(p0, h_temp - (height_in - 1) * stride);
    int p_max = d_min(kernel_size - 1, h_temp);
    int p_start = p_min + ((stride + p0 - (p_min % stride)) % stride);
    int q_min = d_max(q0, w_temp - (width_in - 1) * stride);
    int q_max = d_min(kernel_size - 1, w_temp);
    int q_start = q_min + ((stride + q0 - (q_min % stride)) % stride);
    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;


    // Process all output channels for this spatial position
    for (int out_ch = 0; out_ch < out_channels; out_ch++) {
        float out_val = 0.0f;

        // Group parameters
        int out_channels_per_group = out_channels / groups;
        int in_channels_per_group = in_channels / groups;
        int group = out_ch / out_channels_per_group;
        int out_ch_mod = out_ch % out_channels_per_group;

        // Precompute offset values
        int h_temp = h + padding;
        int w_temp = w + padding;

        // Compute valid kernel ranges
        int p0 = h_temp % stride;
        int q0 = w_temp % stride;

        int p_min = d_max(p0, h_temp - (height_in - 1) * stride);
        int p_max = d_min(kernel_size - 1, h_temp);
        int p_start = p_min + ((stride + p0 - (p_min % stride)) % stride);

        int q_min = d_max(q0, w_temp - (width_in - 1) * stride);
        int q_max = d_min(kernel_size - 1, w_temp);
        int q_start = q_min + ((stride + q0 - (q_min % stride)) % stride);

        // Process input channels in group
        int in_ch_start = group * in_channels_per_group;
        int in_ch_end = in_ch_start + in_channels_per_group;

        for (int in_ch = in_ch_start; in_ch < in_ch_end; in_ch++) {
            for (int p = p_start; p <= p_max; p += stride) {
                int i_in = (h_temp - p) / stride;
                for (int q = q_start; q <= q_max; q += stride) {
                    int j_in = (w_temp - q) / stride;
                    
                    float input_val = input[((b * in_channels + in_ch) * height_in + i_in) * width_in + j_in];
                    float weight_val = weight[((in_ch * out_channels_per_group + out_ch_mod) * kernel_size + p) * kernel_size + q];
                    out_val += input_val * weight_val;
                }
            }
        }

        if (bias_present) {
            out_val += bias[out_ch];
        }

        output[((b * out_channels + out_ch) * height_out + h) * width_out + w] = out_val;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    bool bias_present = bias.has_value();
    if (bias_present) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    int batch = x.size(0);
    int in_channels = x.size(1);
    int height_in = x.size(2);
    int width_in = x.size(3);
    int kernel_size = weight.size(2);
    int out_channels = weight.size(1) * groups;

    int height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int width_out = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch, out_channels, height_out, width_out}, x.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (width_out + block.x - 1) / block.x,
        (height_out + block.y - 1) / block.y,
        batch
    );

    convTranspose2dKernel2DTiling<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_present ? bias.value().data_ptr<float>() : nullptr,
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
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward with 2D tiling (CUDA)");
}