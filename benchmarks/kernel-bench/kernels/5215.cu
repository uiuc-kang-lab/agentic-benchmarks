#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void max_pool1d_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length,
    bool return_indices)
{
    // Calculate the global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = output_length * num_channels * batch_size;

    // Use a loop to cover all required indices based on strides
    for (int idx = tid; idx < total_outputs; idx += gridDim.x * blockDim.x) {
        const int i = idx % output_length;
        const int c = (idx / output_length) % num_channels;
        const int b = idx / (output_length * num_channels);
        
        const int input_start = i * stride - padding;
        float max_val = -INFINITY;
        int max_idx = -1;

        // Calculate the base index for the current batch and channel
        const int base_idx = b * num_channels * input_length + c * input_length;

        // Unrolling loop for kernel size with pragma for potential performance gains
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            const int pos = input_start + k * dilation;
            if (pos >= 0 && pos < input_length) {
                const float val = input[base_idx + pos];
                if (val > max_val) {
                    max_val = val;
                    max_idx = pos;
                }
            }
        }

        // Write the max value to the output
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

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;

    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, 
            options.dtype(torch::kInt64));
    }

    // Optimize thread block size with better occupancy
    const int thread_block_size = 256;

    // Compute blocks needed considering maximum for best efficiency
    const int total_elements = batch_size * num_channels * output_length;
    const int num_blocks = min((total_elements + thread_block_size - 1) / thread_block_size, 32768);

    max_pool1d_fused_kernel<<<num_blocks, thread_block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        num_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        output_length,
        return_indices
    );

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward (CUDA)");
}