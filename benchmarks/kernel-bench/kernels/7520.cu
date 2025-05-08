#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for frequently accessed parameters
__constant__ int c_kernel_size;
__constant__ int c_stride;
__constant__ int c_padding;
__constant__ int c_dims[8];  // [batch, in_channels, out_channels, height_in, width_in, height_out, width_out, groups]

__device__ inline int d_min(int a, int b) { return a < b ? a : b; }
__device__ inline int d_max(int a, int b) { return a > b ? a : b; }

__global__ void convTranspose2dConstantKernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    bool bias_present
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = c_dims[0] * c_dims[2] * c_dims[5] * c_dims[6];  // batch * out_channels * height_out * width_out
    if (idx >= total) return; // Early exit for threads beyond total

    // Coalesced thread mapping: [out_ch, w, h, b]
    int out_ch = idx % c_dims[2];  // out_channels
    int tmp = idx / c_dims[2];
    int w = tmp % c_dims[6];       // width_out
    tmp /= c_dims[6];
    int h = tmp % c_dims[5];       // height_out
    int b = tmp / c_dims[5];

    float out_val = 0.0f;

    int out_channels_per_group = c_dims[2] / c_dims[7];  // out_channels / groups
    int in_channels_per_group = c_dims[1] / c_dims[7];   // in_channels / groups
    int group = out_ch / out_channels_per_group;
    int out_ch_mod = out_ch % out_channels_per_group;

    int h_temp = h + c_padding;
    int w_temp = w + c_padding;

    // Optimized bounds checking
    int p0 = h_temp % c_stride;
    int p_min = d_max(p0, h_temp - (c_dims[3] - 1) * c_stride);  // height_in
    int p_max = d_min(c_kernel_size - 1, h_temp);
    int p_start = p_min + ((p0 - (p_min % c_stride) + c_stride) % c_stride);

    int q0 = w_temp % c_stride;
    int q_min = d_max(q0, w_temp - (c_dims[4] - 1) * c_stride);  // width_in
    int q_max = d_min(c_kernel_size - 1, w_temp);
    int q_start = q_min + ((q0 - (q_min % c_stride) + c_stride) % c_stride);

    int in_ch_start = group * in_channels_per_group;
    int in_ch_end = in_ch_start + in_channels_per_group;

    #pragma unroll 4
    for (int in_ch = in_ch_start; in_ch < in_ch_end; in_ch++) {
        for (int p = p_start; p <= p_max; p += c_stride) {
            int i_in = (h_temp - p) / c_stride;
            for (int q = q_start; q <= q_max; q += c_stride) {
                int j_in = (w_temp - q) / c_stride;
                
                int weight_idx = ((in_ch * out_channels_per_group + out_ch_mod) * c_kernel_size + p) * c_kernel_size + q;
                int input_idx = ((b * c_dims[1] + in_ch) * c_dims[3] + i_in) * c_dims[4] + j_in;
                
                out_val = __fmaf_rn(input[input_idx], weight[weight_idx], out_val);
            }
        }
    }

    if (bias_present) {
        out_val += bias[out_ch];
    }

    int output_idx = ((b * c_dims[2] + out_ch) * c_dims[5] + h) * c_dims[6] + w;
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

    int batch = x.size(0);
    int in_channels = x.size(1);
    int height_in = x.size(2);
    int width_in = x.size(3);
    int kernel_size = weight.size(2);
    int out_channels = weight.size(1) * groups;
    int height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int width_out = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Copy constant parameters to device
    cudaMemcpyToSymbol(c_kernel_size, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(c_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(c_padding, &padding, sizeof(int));
    
    int dims[8] = {batch, in_channels, out_channels, height_in, width_in, height_out, width_out, groups};
    cudaMemcpyToSymbol(c_dims, dims, 8 * sizeof(int));

    auto output = torch::zeros({batch, out_channels, height_out, width_out}, x.options());

    int total_threads = batch * out_channels * height_out * width_out;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    convTranspose2dConstantKernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        bias.has_value()
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d with constant memory optimization");
}