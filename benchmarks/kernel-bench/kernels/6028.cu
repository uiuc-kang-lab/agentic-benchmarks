#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

__global__ void avg_pool1d_kernel(
    const float * __restrict__ input,
    float * __restrict__ output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    // Shared memory for input data
    extern __shared__ float shared_input[];
    
    const int o = threadIdx.x + blockIdx.x * blockDim.x;
    const int channel = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (o >= output_length || channel >= in_channels || batch >= batch_size) 
        return;

    // Calculate the window range for this thread block
    const int block_start = blockIdx.x * blockDim.x * stride - padding;
    const int block_end = (blockIdx.x * blockDim.x + blockDim.x - 1) * stride + kernel_size - padding;
    const int shared_mem_size = block_end - block_start;
    
    // Calculate base indices for input
    const int input_batch_offset = batch * in_channels * input_length;
    const int input_channel_offset = channel * input_length;
    const int base_input_idx = input_batch_offset + input_channel_offset;
    
    // Load data into shared memory
    const int shared_idx_offset = threadIdx.y * shared_mem_size;
    #pragma unroll 4
    for (int i = threadIdx.x; i < shared_mem_size; i += blockDim.x) {
        int global_idx = block_start + i;
        if (global_idx >= 0 && global_idx < input_length) {
            shared_input[shared_idx_offset + i] = __ldg(&input[base_input_idx + global_idx]);
        } else {
            shared_input[shared_idx_offset + i] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Compute average pooling using shared memory
    const int window_start = o * stride - padding - block_start;
    float sum = 0.0f;
    
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int shared_idx = window_start + k;
        if (shared_idx >= 0 && shared_idx < shared_mem_size) {
            sum += shared_input[shared_idx_offset + shared_idx];
        }
    }
    
    // Write result
    const int output_idx = batch * in_channels * output_length + 
                          channel * output_length + o;
    output[output_idx] = sum / kernel_size;
}

torch::Tensor avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {
    
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Optimize thread configuration for aligned access
    dim3 threads(32, 8, 4);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        (in_channels + threads.y - 1) / threads.y,
        (batch_size + threads.z - 1) / threads.z
    );

    avg_pool1d_kernel<<<grid, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward (CUDA)");
}