#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <int BLOCK_SIZE = 256, int TILE_SIZE = 16>
__global__ void convTranspose2dSharedKernel(
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
    extern __shared__ float shared_mem[];
    float* shared_weight = shared_mem;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_channels * height_out * width_out) return;

    // Coalesced thread mapping
    int out_ch = idx % out_channels;
    int tmp = idx / out_channels;
    int w = tmp % width_out;
    tmp /= width_out;
    int h = tmp % height_out;
    int b = tmp / height_out;

    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = out_ch / out_channels_per_group;
    int out_ch_mod = out_ch % out_channels_per_group;

    float out_val = 0.0f;
    
    // Pre-compute spatial offsets
    int h_temp = h + padding;
    int w_temp = w + padding;

    // Compute valid ranges for kernel positions
    int p0 = h_temp % stride;
    int p_min = max(p0, h_temp - (height_in - 1) * stride);
    int p_max = min(kernel_size - 1, h_temp);
    int p_start = p_min + ((p0 - (p_min % stride) + stride) % stride);

    int q0 = w_temp % stride;
    int q_min = max(q0, w_temp - (width_in - 1) * stride);
    int q_max = min(kernel_size - 1, w_temp);
    int q_start = q_min + ((q0 - (q_min % stride) + stride) % stride);

    // Process input channels in tiles
    int in_ch_start = group * in_channels_per_group;
    int in_ch_end = in_ch_start + in_channels_per_group;
    
    for (int in_ch_base = in_ch_start; in_ch_base < in_ch_end; in_ch_base += TILE_SIZE) {
        int valid_channels = min(TILE_SIZE, in_ch_end - in_ch_base);

        // Load weights into shared memory
        if (threadIdx.x < valid_channels * kernel_size) {
            int k_idx = threadIdx.x;
            int in_ch_offset = k_idx / (kernel_size * kernel_size);
            int k_offset = k_idx % (kernel_size * kernel_size);
            int in_ch = in_ch_base + in_ch_offset;
            
            if (in_ch < in_ch_end) {
                int weight_idx = ((in_ch * out_channels_per_group + out_ch_mod) * kernel_size * kernel_size) + k_offset;
                shared_weight[k_idx] = weight[weight_idx];
            }
        }
        __syncthreads();

        for (int p = p_start; p <= p_max; p += stride) {
            int i_in = (h_temp - p) / stride;
            if (i_in < 0 || i_in >= height_in) continue;

            for (int q = q_start; q <= q_max; q += stride) {
                int j_in = (w_temp - q) / stride;
                if (j_in < 0 || j_in >= width_in) continue;

                // Load input tile into shared memory
                if (threadIdx.x < valid_channels) {
                    int in_ch = in_ch_base + threadIdx.x;
                    if (in_ch < in_ch_end) {
                        int input_idx = ((b * in_channels + in_ch) * height_in + i_in) * width_in + j_in;
                        shared_input[threadIdx.x] = input[input_idx];
                    }
                }
                __syncthreads();

                // Compute partial sum for this tile
                for (int c = 0; c < valid_channels; c++) {
                    int weight_idx = (c * kernel_size + p) * kernel_size + q;
                    out_val = __fmaf_rn(shared_input[c], shared_weight[weight_idx], out_val);
                }
                __syncthreads();
            }
        }
    }

    if (bias_present) {
        out_val += bias[out_ch];
    }

    // Coalesced write to output
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

    constexpr int BLOCK_SIZE = 256;
    constexpr int TILE_SIZE = 16;
    const int total_threads = batch * out_channels * height_out * width_out;
    const int grid_size = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Shared memory size: space for input tile and weight tile
    const int shared_mem_size = (TILE_SIZE * TILE_SIZE + TILE_SIZE * kernel_size * kernel_size) * sizeof(float);

    convTranspose2dSharedKernel<BLOCK_SIZE, TILE_SIZE><<<grid_size, BLOCK_SIZE, shared_mem_size>>>(
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
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d with shared memory optimization");
}