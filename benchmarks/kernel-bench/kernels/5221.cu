#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <int BLOCK_SIZE>
__global__ void shared_mem_pool1d_kernel(
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
    
    const int tid = threadIdx.x;
    const int out_idx = blockIdx.x * BLOCK_SIZE + tid;
    const int total_outputs = output_length * num_channels * batch_size;
    
    if (out_idx >= total_outputs) return;
    
    const int o = out_idx % output_length;
    const int c = (out_idx / output_length) % num_channels;
    const int b = out_idx / (output_length * num_channels);
    
    const int input_start = o * stride - padding;
    const int base_idx = b * num_channels * input_length + c * input_length;
    
    float max_val = -INFINITY;
    int max_idx = -1;
    
    // Load input data into shared memory
    const int sh_mem_size = BLOCK_SIZE + kernel_size - 1;
    const int sh_start = input_start - (tid % stride);
    
    if (tid < sh_mem_size) {
        const int global_idx = base_idx + sh_start + tid;
        if (sh_start + tid >= 0 && sh_start + tid < input_length) {
            shared_input[tid] = input[global_idx];
        } else {
            shared_input[tid] = -INFINITY;
        }
    }
    
    __syncthreads();  // Single sync after shared memory load
    
    // Compute max pooling using shared memory
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        const int pos = tid + k;
        if (pos >= 0 && pos < sh_mem_size) {
            const float val = shared_input[pos];
            if (val > max_val) {
                max_val = val;
                max_idx = input_start + k * dilation;
            }
        }
    }
    
    if (out_idx < total_outputs) {
        output[out_idx] = max_val;
        if (return_indices && max_idx >= 0) {
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

    constexpr int BLOCK_SIZE = 256;
    const int total_elements = batch_size * num_channels * output_length;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Calculate shared memory size
    const int shared_mem_size = (BLOCK_SIZE + kernel_size - 1) * sizeof(float);

    shared_mem_pool1d_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
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