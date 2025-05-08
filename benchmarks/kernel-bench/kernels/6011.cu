#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    // Shared memory for input tile
    extern __shared__ float shared_input[];
    
    const int o = threadIdx.x + blockIdx.x * blockDim.x;
    const int channel = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = threadIdx.z + blockIdx.z * blockDim.z;
    
    // Local thread index for shared memory operations
    const int local_idx = threadIdx.x + threadIdx.y * blockDim.x;
    const int tile_size = blockDim.x * stride + kernel_size - stride;
    
    // Early exit for threads outside bounds
    if (channel >= in_channels || batch >= batch_size) return;
    
    // Pre-compute input base offset for this thread
    const int input_batch_offset = batch * in_channels * input_length;
    const int input_channel_offset = channel * input_length;
    const int base_idx = input_batch_offset + input_channel_offset;
    
    // Load input tile into shared memory
    const int tile_start = blockIdx.x * blockDim.x * stride - padding;
    const int shared_mem_size = tile_size;
    
    // Collaborative loading of input data into shared memory
    for (int i = local_idx; i < shared_mem_size; i += blockDim.x * blockDim.y) {
        int global_idx = tile_start + i;
        shared_input[i] = (global_idx >= 0 && global_idx < input_length) 
                         ? input[base_idx + global_idx] 
                         : 0.0f;
    }
    
    __syncthreads();
    
    // Process only if this thread has a valid output position
    if (o < output_length) {
        float sum = 0.0f;
        
        // Calculate position in shared memory
        const int local_start = (o - blockIdx.x * blockDim.x) * stride;
        
        // Main computation loop using shared memory
        #pragma unroll 4
        for (int k = 0; k < kernel_size; ++k) {
            sum += shared_input[local_start + k];
        }
        
        // Compute output index
        const int output_idx = batch * in_channels * output_length + 
                             channel * output_length + o;
        
        // Write result
        output[output_idx] = sum / kernel_size;
    }
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

    // Optimize thread block dimensions for better occupancy
    // and maintain good memory access patterns
    dim3 threads(32, 8, 4);  // 32 threads per warp, utilizing 3D blocking
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