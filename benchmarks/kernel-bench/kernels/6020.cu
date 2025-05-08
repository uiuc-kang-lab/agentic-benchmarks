#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for frequently accessed parameters
__constant__ int pool_params[5];  // [kernel_size, stride, padding, input_length, output_length]

// Shared memory tile size
#define TILE_SIZE 32

__global__ void optimized_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels) {
    
    // Define shared memory
    extern __shared__ float shared_input[];  // Extra space for kernel overlap
    
    const int kernel_size = pool_params[0];
    const int stride = pool_params[1];
    const int padding = pool_params[2];
    const int input_length = pool_params[3];
    const int output_length = pool_params[4];
    
    const int tid = threadIdx.x;
    const int o = blockIdx.x * TILE_SIZE + tid;
    const int channel = blockIdx.y;
    const int batch = blockIdx.z;
    
    if (channel >= in_channels || batch >= batch_size) return;
    
    // Calculate input indices for this tile
    const int tile_start = blockIdx.x * TILE_SIZE * stride - padding;
    const int tile_elements = TILE_SIZE * stride + kernel_size;
    
    // Load input data into shared memory
    for (int i = tid; i < tile_elements && (tile_start + i) < input_length; i += TILE_SIZE) {
        if (tile_start + i >= 0) {
            shared_input[i] = input[batch * in_channels * input_length + 
                                  channel * input_length + 
                                  tile_start + i];
        } else {
            shared_input[i] = 0.0f;
        }
    }
    __syncthreads();
    
    // Compute average pooling
    if (o < output_length) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            const int pos_local = tid * stride + k;
            sum += shared_input[pos_local];
        }
        
        output[batch * in_channels * output_length + 
               channel * output_length + o] = sum / kernel_size;
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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    // Copy parameters to constant memory
    int h_pool_params[5] = {kernel_size, stride, padding, input_length, output_length};
    cudaMemcpyToSymbol(pool_params, h_pool_params, 5 * sizeof(int));

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    dim3 threads(TILE_SIZE);
    dim3 grid(
        (output_length + TILE_SIZE - 1) / TILE_SIZE,
        in_channels,
        batch_size
    );

    optimized_avg_pool1d_kernel<<<grid, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "Optimized 1D Average Pooling forward (CUDA)");
}