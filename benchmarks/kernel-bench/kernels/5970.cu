#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constant memory for frequently accessed read-only data
__constant__ int const_kernel_size;
__constant__ int const_stride;
__constant__ int const_padding;

__global__ void constant_memory_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int input_length,
    const int output_length,
    const int batch_size,
    const int in_channels) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * in_channels * output_length;
    
    if (idx >= total_elements) return;
    
    const int o = idx % output_length;
    const int channel = (idx / output_length) % in_channels;
    const int batch = idx / (output_length * in_channels);
    
    const int input_batch_offset = batch * in_channels * input_length;
    const int input_channel_offset = channel * input_length;
    const int input_base = input_batch_offset + input_channel_offset;
    
    float sum = 0.0f;
    const int start_idx = o * const_stride - const_padding;
    
    #pragma unroll
    for (int k = 0; k < const_kernel_size; ++k) {
        const int pos_input = start_idx + k;
        if (pos_input >= 0 && pos_input < input_length) {
            sum += input[input_base + pos_input];
        }
    }
    
    output[idx] = sum / const_kernel_size;
}

torch::Tensor constant_memory_avg_pool1d_forward(
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
    
    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Copy frequently accessed parameters to constant memory
    cudaMemcpyToSymbol(const_kernel_size, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(const_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(const_padding, &padding, sizeof(int));
    
    const int total_elements = batch_size * in_channels * output_length;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    constant_memory_avg_pool1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        input_length,
        output_length,
        batch_size,
        in_channels
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &constant_memory_avg_pool1d_forward, "1D Average Pooling forward (CUDA) with constant memory");
}
