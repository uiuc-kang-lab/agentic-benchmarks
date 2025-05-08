#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void max_pool1d_flattened_kernel(
    const float* __restrict__ input,
    float* output,
    int64_t* indices,
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
    const int total_threads = batch_size * num_channels * output_length;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_threads) return;
    
    const int b = tid / (num_channels * output_length);
    const int remaining = tid % (num_channels * output_length);
    const int c = remaining / output_length;
    const int i = remaining % output_length;
    
    const int input_start = i * stride - padding;
    float max_val = -INFINITY;
    int max_idx = -1;

    for (int k = 0; k < kernel_size; ++k) {
        const int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            const float val = input[b * num_channels * input_length + c * input_length + pos];
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

    max_pool1d_flattened_kernel<<<num_blocks, threads_per_block>>>(
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