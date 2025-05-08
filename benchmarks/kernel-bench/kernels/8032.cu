#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MAX_WEIGHT_SIZE 4096
#define BLOCK_SIZE 256
#define SHARED_MEM_SIZE 2048

__constant__ float c_weight[MAX_WEIGHT_SIZE];

__global__ void warp_aligned_conv1d_kernel(
    const float* __restrict__ input,
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
    
    extern __shared__ float shared_input[];
    
    // Calculate global position
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Each block handles one batch and one output channel
    const int b = blockIdx.x;
    const int o = blockIdx.y;
    
    // Calculate group information
    const int group_size_out = out_channels / groups;
    const int group_in_channels = in_channels / groups;
    const int g = o / group_size_out;
    const int group_start = g * group_in_channels;
    
    // Base output position for this warp
    const int base_out_pos = blockIdx.z * BLOCK_SIZE + tid;
    
    if (base_out_pos < output_width) {
        float sum = 0.0f;
        
        // Process input channels in chunks to fit in shared memory
        for (int ic_chunk = 0; ic_chunk < group_in_channels; ic_chunk++) {
            const int ic = group_start + ic_chunk;
            
            // Load input data into shared memory
            const int input_load_pos = base_out_pos * stride - padding;
            if (input_load_pos >= 0 && input_load_pos < input_width) {
                shared_input[tid] = input[b * in_channels * input_width + 
                                        ic * input_width + 
                                        input_load_pos];
            } else {
                shared_input[tid] = 0.0f;
            }
            __syncthreads();
            
            // Compute convolution for this input channel
            #pragma unroll
            for (int k = 0; k < kernel_size; k++) {
                const int input_pos = base_out_pos + padding - k;
                if (input_pos % stride == 0) {
                    const int i = input_pos / stride;
                    if (i >= 0 && i < input_width) {
                        const int weight_idx = (ic * group_size_out + (o % group_size_out)) * kernel_size + k;
                        sum += shared_input[tid - k] * c_weight[weight_idx];
                    }
                }
            }
            __syncthreads();
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += bias[o];
        }
        
        // Write result to global memory - ensures coalesced writes as consecutive threads
        // write to consecutive memory locations
        const int out_idx = b * out_channels * output_width + 
                          o * output_width + 
                          base_out_pos;
        output[out_idx] = sum;
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
    CHECK_CONTIGUOUS(x);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(weight);
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_width = x.size(2);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(0) * groups;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());
    
    // Copy weights to constant memory
    TORCH_CHECK(weight.numel() <= MAX_WEIGHT_SIZE, "Weight size exceeds constant memory limit");
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), 
                       weight.numel() * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    
    // Calculate grid and block dimensions
    const int blocks_per_output = (output_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 grid(batch_size, out_channels, blocks_per_output);
    dim3 block(BLOCK_SIZE);
    
    const int shared_mem_size = SHARED_MEM_SIZE * sizeof(float);
    
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    warp_aligned_conv1d_kernel<<<grid, block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
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
    m.def("forward", &forward, "Warp-aligned transposed 1D convolution forward (CUDA)");
}