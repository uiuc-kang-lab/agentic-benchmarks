#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Structure to hold pooling parameters in constant memory
struct PoolParams {
    int kernel_size;
    int stride;
    int padding;
    int input_length;
    int output_length;
};

// Declare constant memory for pooling parameters (fits within hardware limits)
__constant__ PoolParams cPoolParams;

// Kernel that uses constant memory for frequently accessed read-only parameters
__global__ void avg_pool1d_kernel_const(
    const float *input,
    float *output,
    int batch_size,
    int in_channels) {
    
    // Shared memory for input data
    extern __shared__ float shared_input[];
    
    // 3D thread mapping: each thread computes one output element
    int o = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = threadIdx.y + blockIdx.y * blockDim.y;
    int batch = threadIdx.z + blockIdx.z * blockDim.z;
    
    // Calculate the shared memory offset for this thread block
    const int BLOCK_SIZE = blockDim.x;
    const int shared_offset = threadIdx.y * (BLOCK_SIZE + cPoolParams.kernel_size - 1);
    
    // Load input data into shared memory with padding
    if (channel < in_channels && batch < batch_size) {
        const int base_idx = batch * (in_channels * cPoolParams.input_length) + channel * cPoolParams.input_length;
        
        // Each thread loads its corresponding input elements
        for (int i = threadIdx.x; i < BLOCK_SIZE + cPoolParams.kernel_size - 1; i += BLOCK_SIZE) {
            int pos_input = blockIdx.x * BLOCK_SIZE + i - cPoolParams.padding;
            if (pos_input >= 0 && pos_input < cPoolParams.input_length) {
                shared_input[shared_offset + i] = input[base_idx + pos_input];
            } else {
                shared_input[shared_offset + i] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (o >= cPoolParams.output_length || channel >= in_channels || batch >= batch_size) return;
    
    float sum = 0.0f;
    // Accumulate over the pooling window using shared memory
    const int start_idx = threadIdx.x * cPoolParams.stride;
    for (int k = 0; k < cPoolParams.kernel_size; ++k) {
        sum += shared_input[shared_offset + start_idx + k];
    }
    
    int output_idx = batch * (in_channels * cPoolParams.output_length) + channel * cPoolParams.output_length + o;
    output[output_idx] = sum / cPoolParams.kernel_size;
}

// Forward function copying pooling parameters to constant memory and launching the kernel
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

    // Copy pooling parameters to constant memory (read-only and frequently accessed)
    PoolParams h_params;
    h_params.kernel_size = kernel_size;
    h_params.stride = stride;
    h_params.padding = padding;
    h_params.input_length = input_length;
    h_params.output_length = output_length;
    cudaMemcpyToSymbol(cPoolParams, &h_params, sizeof(PoolParams));

    // Configure 3D grid and block dimensions
    dim3 threads(32, 8, 4);
    dim3 blocks(
        (output_length + threads.x - 1) / threads.x,
        (in_channels + threads.y - 1) / threads.y,
        (batch_size + threads.z - 1) / threads.z
    );

    avg_pool1d_kernel_const<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward using constant memory for pooling parameters (CUDA)");
}
