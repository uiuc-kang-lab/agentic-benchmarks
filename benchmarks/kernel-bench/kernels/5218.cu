#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined optimized kernel using flat index and loop unrolling

__global__ void max_pool1d_kernel_combined(
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
    const bool return_indices) {

    // Total number of output elements (batch * channels * output_length)
    const int total_outputs = batch_size * num_channels * output_length;
    
    // Each thread processes one or more output elements using a flat index
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_outputs; 
         idx += gridDim.x * blockDim.x) {

        // Decode flat index into batch (b), channel (c) and output index (i)
        const int i = idx % output_length;
        const int c = (idx / output_length) % num_channels;
        const int b = idx / (output_length * num_channels);

        // Calculate the start index in the input
        const int input_start = i * stride - padding;

        float max_val = -INFINITY;
        int max_idx = -1;
        
        // Base index for the current batch and channel
        const int base_idx = b * num_channels * input_length + c * input_length;

        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            const int pos = input_start + k * dilation;
            if (pos >= 0 && pos < input_length) {
                float val = input[base_idx + pos];
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

// Forward wrapper for PyTorch

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {

    TORCH_CHECK(x.dim() == 3, "Input must be 3D: [batch, channels, length]");
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    // Compute output length
    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    // Flattened total number of output elements
    const int total_elements = batch_size * num_channels * output_length;
    // Use an optimized block size for modern GPUs
    const int block_size = 256;
    // Limit the max number of blocks (e.g., for NVIDIA H100 architecture)
    const int max_blocks = 32768;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    num_blocks = (num_blocks > max_blocks) ? max_blocks : num_blocks;

    max_pool1d_kernel_combined<<<num_blocks, block_size>>>(
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

    // If requested, concatenate indices along the last dimension
    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized MaxPool1D forward (CUDA)");
}
