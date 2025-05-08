#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define TILE_SIZE 128

__global__ void aligned_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding) {
    
    __shared__ float shared_weight[TILE_SIZE];
    
    const int output_length = (input_length - 1) * stride - 2 * padding + 
                            kernel_size + output_padding;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;
    const int out_ch = blockIdx.z;
    
    // Pre-calculate offsets for aligned access
    const int weight_offset = out_ch * in_channels * kernel_size;
    const int batch_offset = batch_idx * in_channels * input_length;
    const int output_offset = (batch_idx * out_channels + out_ch) * output_length;
    
    // Process multiple elements per thread for better efficiency
    for (int out_pos = tid; out_pos < output_length; out_pos += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        
        // Calculate valid input positions
        const int start_k = max(0, (padding - out_pos + stride - 1) / stride);
        const int end_k = min(kernel_size, (padding - out_pos + input_length * stride) / stride);
        
        // Load weights into shared memory
        for (int k = threadIdx.x; k < kernel_size && k < TILE_SIZE; k += blockDim.x) {
            shared_weight[k] = __ldg(&weight[weight_offset + k]);
        }
        __syncthreads();
        
        // Main convolution loop with aligned reads
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            const int in_ch_offset = batch_offset + in_ch * input_length;
            
            #pragma unroll 4
            for (int k = start_k; k < end_k; ++k) {
                const int in_pos = (out_pos + padding - k) / stride;
                if ((out_pos + padding - k) % stride == 0) {
                    float in_val = __ldg(&input[in_ch_offset + in_pos]);
                    sum += in_val * shared_weight[k];
                }
            }
        }
        
        // Write output
        output[output_offset + out_pos] = sum;
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
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    
    const auto input_sizes = x.sizes();
    const int batch_size = input_sizes[0];
    const int in_channels = input_sizes[1];
    const int input_length = input_sizes[2];
    
    const auto weight_sizes = weight.sizes();
    const int out_channels = weight_sizes[1];
    const int kernel_size = weight_sizes[2];
    
    const int output_length = (input_length - 1) * stride - 2 * padding + 
                            kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());
    
    const int threads = 256;
    const int blocks_x = (output_length + threads - 1) / threads;
    
    dim3 grid(blocks_x, batch_size, out_channels);
    dim3 block(threads);
    
    aligned_conv_transpose_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_padding
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Aligned memory transposed 1D convolution forward (CUDA)");
}