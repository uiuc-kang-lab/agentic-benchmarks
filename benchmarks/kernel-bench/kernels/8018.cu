#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define ALIGN_BYTES 16  // 128-bit alignment
#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 4

__forceinline__ __device__ float4 load_float4(const float* addr) {
    float4 result;
    result = *reinterpret_cast<const float4*>(addr);
    return result;
}

__global__ void aligned_conv_transposed1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_width,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int groups) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_idx = bid * blockDim.x * ELEMENTS_PER_THREAD + tid;
    
    // Pre-compute group parameters
    const int channels_per_group = in_channels / groups;
    const int out_channels_per_group = out_channels / groups;
    
    // Process ELEMENTS_PER_THREAD elements per thread
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = global_idx + i * blockDim.x;
        if (idx >= batch_size * out_channels * output_width) return;
        
        // Calculate position indices
        const int pos = idx % output_width;
        const int oc = (idx / output_width) % out_channels;
        const int batch = idx / (output_width * out_channels);
        
        // Calculate group
        const int group = oc / out_channels_per_group;
        const int oc_within_group = oc % out_channels_per_group;
        
        float sum = 0.0f;
        
        // Compute valid input range for this output position
        const int input_start = (pos + padding) / stride;
        const int input_end = min((pos + padding + kernel_size) / stride, input_width);
        
        // Process input channels in aligned chunks where possible
        const int aligned_channels = channels_per_group & ~3;  // Round down to multiple of 4
        
        for (int in_pos = input_start; in_pos < input_end; in_pos++) {
            const int k = pos + padding - in_pos * stride;
            if (k < 0 || k >= kernel_size) continue;
            
            // Base indices for current position
            const int input_batch_offset = batch * in_channels * input_width;
            const int group_channel_offset = group * channels_per_group;
            
            // Process aligned channels using float4
            for (int ic = 0; ic < aligned_channels; ic += 4) {
                const int real_ic = group_channel_offset + ic;
                const float4 input_vec = load_float4(&input[input_batch_offset + real_ic * input_width + in_pos]);
                
                // Load weights using __ldg for better caching
                const float4 weight_vec = load_float4(&weight[(real_ic * out_channels_per_group + oc_within_group) * kernel_size + k]);
                
                sum += input_vec.x * __ldg(&weight_vec.x);
                sum += input_vec.y * __ldg(&weight_vec.y);
                sum += input_vec.z * __ldg(&weight_vec.z);
                sum += input_vec.w * __ldg(&weight_vec.w);
            }
            
            // Handle remaining channels
            for (int ic = aligned_channels; ic < channels_per_group; ic++) {
                const int real_ic = group_channel_offset + ic;
                const float input_val = __ldg(&input[input_batch_offset + real_ic * input_width + in_pos]);
                const float weight_val = __ldg(&weight[(real_ic * out_channels_per_group + oc_within_group) * kernel_size + k]);
                sum += input_val * weight_val;
            }
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += __ldg(&bias[oc]);
        }
        
        // Write output
        output[idx] = sum;
    }
}

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    CHECK_CUDA(x);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(weight);
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_width = x.size(2);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1) * groups;
    
    // Calculate output width
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());
    
    const int total_elements = batch_size * out_channels * output_width;
    const int total_threads = (total_elements + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    const int num_blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        CHECK_CUDA(bias.value());
        CHECK_CONTIGUOUS(bias.value());
        bias_ptr = bias.value().data_ptr<float>();
    }
    
    aligned_conv_transposed1d_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_width,
        output_width,
        kernel_size,
        stride,
        padding,
        groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized transposed 1D convolution forward (CUDA)");
}