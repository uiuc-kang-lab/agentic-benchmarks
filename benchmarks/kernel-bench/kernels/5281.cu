#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void max_pool1d_shared_kernel(
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
    extern __shared__ float shared_input[];
    
    const int total_threads = batch_size * num_channels * output_length;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_threads) return;
    
    // Unified batch/channel/output index calculation
    const int b = tid / (num_channels * output_length);
    const int remaining = tid % (num_channels * output_length);
    const int c = remaining / output_length;
    const int i = remaining % output_length;
    
    // Calculate input window
    const int input_start = i * stride - padding + (kernel_size - 1) * dilation;
    
    // Preload input into shared memory
    const int block_start = threadIdx.x / output_length * input_length;
    const int load_size = max(0, block_start + input_length) - block_start;
    if (threadIdx.x < load_size) {
        shared_input[threadIdx.x] = input[b * num_channels * input_length + c * input_length + block_start + threadIdx.x];
    }
    __syncthreads();
    
    // Compute maximum using shared memory
    float max_val = -INFINITY;
    int max_idx = -1;
    for (int k = 0; k < kernel_size; ++k) {
        const int pos = input_start + k * dilation;
        if (pos >= block_start && pos < block_start + input_length) {
            const float val = shared_input[pos - block_start];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }
    
    const int out_idx = b * num_channels * output_length + c * output_length + i;
    output[out_idx] = max_val;
    if (return_indices) indices[out_idx] = max_idx;
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

    const int total_elements = batch_size * num_channels * output_length;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    const int shared_mem_size = input_length * sizeof(float);

    max_pool1d_shared_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
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
    m.def("forward", &forward, "MaxPool1D forward with shared memory (CUDA)");
}