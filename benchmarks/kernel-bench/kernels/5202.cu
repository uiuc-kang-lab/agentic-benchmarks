#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized device function for max pooling operation
__device__ __forceinline__ void perform_max_pooling(
    const float* __restrict__ input,
    const int base_idx,
    const int input_start,
    const int input_length,
    const int kernel_size,
    const int dilation,
    float &max_val,
    int &max_idx
) {
    max_val = -INFINITY;
    max_idx = -1;
    
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
}

__global__ void hybrid_maxpool1d_kernel(
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
    bool return_indices
) {
    // Use 2D grid for better occupancy
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int channel_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    // Early exit if out of bounds
    if (batch_idx >= batch_size || channel_idx >= num_channels) return;

    // Compute base index for current batch and channel
    const int base_idx = batch_idx * num_channels * input_length + channel_idx * input_length;
    
    // Each thread processes multiple elements along the output length
    for (int i = tid + bid * blockDim.x; i < output_length; i += blockDim.x * gridDim.x) {
        const int input_start = i * stride - padding;
        float max_val;
        int max_idx;
        
        perform_max_pooling(
            input,
            base_idx,
            input_start,
            input_length,
            kernel_size,
            dilation,
            max_val,
            max_idx
        );

        const int out_idx = batch_idx * num_channels * output_length + 
                           channel_idx * output_length + i;
        output[out_idx] = max_val;
        if (return_indices) {
            indices[out_idx] = max_idx;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices
) {
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

    // Optimize thread and block configuration
    const int thread_block_size = 256;
    const int max_blocks_x = 32;
    const dim3 blocks(
        min(max_blocks_x, (output_length + thread_block_size - 1) / thread_block_size),
        num_channels,
        batch_size
    );
    const dim3 threads(thread_block_size, 1, 1);

    hybrid_maxpool1d_kernel<<<blocks, threads>>>(
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