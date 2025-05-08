#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute padding masks for a row to avoid branching in inner loop
__device__ __forceinline__ unsigned int compute_mask(
    int start_idx,
    int valid_start,
    int valid_end,
    int warp_size = 32
) {
    unsigned int mask = 0;
    #pragma unroll
    for (int i = 0; i < warp_size; i++) {
        int idx = start_idx + i;
        if (idx >= valid_start && idx < valid_end) {
            mask |= (1u << i);
        }
    }
    return mask;
}

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    // Calculate position
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    
    // Each warp processes a row of output
    const int total_rows = batch_size * out_channels * out_h;
    if (global_warp_id >= total_rows) return;
    
    // Decode position
    int tmp = global_warp_id;
    const int h_out = tmp % out_h;
    tmp /= out_h;
    const int c_out = tmp % out_channels;
    const int b = tmp / out_channels;
    
    const int g = c_out / channels_per_group;
    const int m = c_out % channels_per_group;

    // Pre-compute input row positions and validity
    int valid_h_positions[9];  // Assume max kernel height is 9
    bool valid_h[9];
    #pragma unroll
    for (int kh = 0; kh < kernel_h; kh++) {
        int h_in = h_out * stride_h - padding_h + kh * dilation_h;
        valid_h[kh] = (h_in >= 0 && h_in < in_h);
        valid_h_positions[kh] = h_in;
    }

    // Process output elements in a row
    float sum = 0.0f;
    const int base_w = lane_id * stride_w + padding_w;
    
    // Pre-compute input base indices
    const int batch_offset = b * in_channels * in_h * in_w;
    const int channel_offset = g * in_h * in_w;
    
    #pragma unroll
    for (int kh = 0; kh < kernel_h; kh++) {
        if (!valid_h[kh]) continue;
        
        const int h_in = valid_h_positions[kh];
        const int h_offset = h_in * in_w;
        
        #pragma unroll
        for (int kw = 0; kw < kernel_w; kw++) {
            const int w_in = base_w + kw * dilation_w;
            const bool valid_w = (w_in >= 0 && w_in < in_w);
            
            if (valid_w) {
                const int input_idx = batch_offset + channel_offset + h_offset + w_in;
                const int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    // Write output if position is valid
    const int w_out = lane_id;
    if (w_out < out_w) {
        const int out_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
        output[out_idx] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    // Launch one warp per output row
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * 32;
    const int num_rows = batch_size * out_channels * out_h;
    const int num_blocks = (num_rows + warps_per_block - 1) / warps_per_block;

    depthwise_conv2d_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise Conv2D forward (CUDA)");
}