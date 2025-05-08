#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for frequently accessed parameters
__constant__ int c_kernel_size;
__constant__ int c_stride;
__constant__ int c_padding;
__constant__ int c_dilation;
__constant__ int c_input_length;
__constant__ int c_output_length;
__constant__ int c_num_channels;

__global__ void constant_mem_max_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    bool return_indices)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int total_outputs = c_output_length * c_num_channels * batch_size;
    
    for (int idx = tid; idx < total_outputs; idx += blockDim.x * gridDim.x) {
        const int i = idx % c_output_length;
        const int c = (idx / c_output_length) % c_num_channels;
        const int b = idx / (c_output_length * c_num_channels);
        
        const int input_start = i * c_stride - c_padding;
        float max_val = -INFINITY;
        int max_idx = -1;

        const int base_idx = b * c_num_channels * c_input_length + c * c_input_length;
        
        #pragma unroll
        for (int k = 0; k < c_kernel_size; ++k) {
            const int pos = input_start + k * c_dilation;
            if (pos >= 0 && pos < c_input_length) {
                const float val = input[base_idx + pos];
                if (val > max_val) {
                    max_val = val;
                    max_idx = pos;
                }
            }
        }

        output[idx] = max_val;
        if (return_indices) {
            indices[idx] = max_idx;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices)
{
    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    // Copy constant parameters to device constant memory
    int h_kernel_size = static_cast<int>(kernel_size);
    int h_stride = static_cast<int>(stride);
    int h_padding = static_cast<int>(padding);
    int h_dilation = static_cast<int>(dilation);
    
    cudaMemcpyToSymbol(c_kernel_size, &h_kernel_size, sizeof(int));
    cudaMemcpyToSymbol(c_stride, &h_stride, sizeof(int));
    cudaMemcpyToSymbol(c_padding, &h_padding, sizeof(int));
    cudaMemcpyToSymbol(c_dilation, &h_dilation, sizeof(int));
    cudaMemcpyToSymbol(c_input_length, &input_length, sizeof(int));
    cudaMemcpyToSymbol(c_output_length, &output_length, sizeof(int));
    cudaMemcpyToSymbol(c_num_channels, &num_channels, sizeof(int));

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;

    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, 
            options.dtype(torch::kInt64));
    }

    const int total_elements = batch_size * num_channels * output_length;
    const int thread_block_size = 256;
    const int num_blocks = min(
        (total_elements + thread_block_size - 1) / thread_block_size,
        32768
    );

    constant_mem_max_pool1d_kernel<<<num_blocks, thread_block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        return_indices
    );

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with constant memory optimization (CUDA)");
}