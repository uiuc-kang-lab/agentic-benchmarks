#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for frequently accessed parameters
__constant__ int c_kernel_size;
__constant__ int c_stride;
__constant__ int c_padding;
__constant__ int c_input_length;
__constant__ int c_output_length;
__constant__ int c_batch_size;
__constant__ int c_in_channels;

__global__ void avg_pool1d_kernel(
    const float *__restrict__ input,
    float *output) {

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (o >= c_output_length || channel >= c_in_channels || batch >= c_batch_size) return;

    float sum = 0.0f;
    const float inv_kernel_size = 1.0f / c_kernel_size;
    
    // Calculate input index base for this batch and channel
    const int base_idx = batch * c_in_channels * c_input_length + channel * c_input_length;
    
    // Calculate window boundaries
    const int start_idx = o * c_stride - c_padding;
    const int end_idx = start_idx + c_kernel_size;
    
    // Vectorized load using aligned memory access pattern
    #pragma unroll 4
    for (int k = 0; k < c_kernel_size; ++k) {
        int pos_input = start_idx + k;
        if (pos_input >= 0 && pos_input < c_input_length) {
            sum += input[base_idx + pos_input];
        }
    }

    // Write result with coalesced memory access
    output[batch * c_in_channels * c_output_length + channel * c_output_length + o] = sum * inv_kernel_size;
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

    // Copy parameters to constant memory
    cudaMemcpyToSymbol(c_kernel_size, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(c_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(c_padding, &padding, sizeof(int));
    cudaMemcpyToSymbol(c_input_length, &input_length, sizeof(int));
    cudaMemcpyToSymbol(c_output_length, &output_length, sizeof(int));
    cudaMemcpyToSymbol(c_batch_size, &batch_size, sizeof(int));
    cudaMemcpyToSymbol(c_in_channels, &in_channels, sizeof(int));

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Optimize block size for better occupancy
    const int block_size = 256;
    dim3 threads(block_size);
    dim3 grid(
        (output_length + block_size - 1) / block_size,
        in_channels,
        batch_size
    );

    avg_pool1d_kernel<<<grid, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling with constant memory (CUDA)");
}