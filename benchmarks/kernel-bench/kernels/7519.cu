#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline int d_min(int a, int b) { return a < b ? a : b; }
__device__ inline int d_max(int a, int b) { return a > b ? a : b; }

__global__ void convTranspose2dTunedKernel(
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
    // 2D block configuration (8x16 = 128 threads)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Calculate output position
    const int w = (bx * blockDim.x + tx) % width_out;
    const int h = (by * blockDim.y + ty) % height_out;
    const int out_ch = ((bx * blockDim.x + tx) / width_out) * (blockDim.y) + ((by * blockDim.y + ty) / height_out);
    const int b = (out_ch / out_channels);
    
    if (w >= width_out || h >= height_out || out_ch >= out_channels || b >= batch)
        return;

    float out_val = 0.0f;

    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    const int group = out_ch / out_channels_per_group;
    const int out_ch_mod = out_ch % out_channels_per_group;

    const int h_temp = h + padding;
    const int w_temp = w + padding;

    // Optimized bounds checking
    const int p0 = h_temp % stride;
    const int p_min = d_max(p0, h_temp - (height_in - 1) * stride);
    const int p_max = d_min(kernel_size - 1, h_temp);
    const int p_start = p_min + ((p0 - p_min + stride) % stride);

    const int q0 = w_temp % stride;
    const int q_min = d_max(q0, w_temp - (width_in - 1) * stride);
    const int q_max = d_min(kernel_size - 1, w_temp);
    const int q_start = q_min + ((q0 - (q_min % stride) + stride) % stride);

    const int in_ch_start = group * in_channels_per_group;
    const int in_ch_end = in_ch_start + in_channels_per_group;

    #pragma unroll 4
    for (int in_ch = in_ch_start; in_ch < in_ch_end; in_ch++) {
        for (int p = p_start; p <= p_max; p += stride) {
            const int i_in = (h_temp - p) / stride;
            for (int q = q_start; q <= q_max; q += stride) {
                const int j_in = (w_temp - q) / stride;
                
                const int weight_idx = ((in_ch * out_channels_per_group + out_ch_mod) * kernel_size + p) * kernel_size + q;
                const int input_idx = ((b * in_channels + in_ch) * height_in + i_in) * width_in + j_in;
                
                out_val = __fmaf_rn(input[input_idx], weight[weight_idx], out_val);
            }
        }
    }

    if (bias_present) {
        out_val += bias[out_ch];
    }

    const int output_idx = ((b * out_channels + out_ch) * height_out + h) * width_out + w;
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
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "Inputs must be contiguous");

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int height_in = x.size(2);
    const int width_in = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1) * groups;

    const int height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int width_out = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch, out_channels, height_out, width_out}, x.options());

    // 2D block configuration: 8x16 threads = 128 threads per block
    dim3 threads(8, 16);
    dim3 blocks(
        (width_out + threads.x - 1) / threads.x,
        (height_out + threads.y - 1) / threads.y
    );

    convTranspose2dTunedKernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        height_in, width_in,
        kernel_size, stride, padding,
        height_out, width_out,
        groups, bias.has_value()
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d with tuned block size");
}