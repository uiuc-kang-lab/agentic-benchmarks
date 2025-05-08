#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void max_pool1d_kernel(
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
    const bool return_indices)
{
    // Calculate global position
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (out_idx >= output_length || c >= num_channels || b >= batch_size) return;

    // Calculate input and output offsets
    const int batch_offset = b * num_channels * input_length;
    const int channel_offset = c * input_length;
    const int input_offset = batch_offset + channel_offset;
    
    const int out_offset = b * num_channels * output_length + c * output_length + out_idx;
    const int input_start = out_idx * stride - padding;

    float max_val = -FLT_MAX;
    int max_idx = -1;

    #pragma unroll
    for (int k = 0; k < kernel_size; k++) {
        const int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            const float val = input[input_offset + pos];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }

    output[out_offset] = max_val;
    if (return_indices) {
        indices[out_offset] = max_idx;
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

    // Optimize thread block configuration
    const dim3 threads(32, 8);
    const dim3 blocks(
        (output_length + threads.x - 1) / threads.x,
        (num_channels + threads.y - 1) / threads.y,
        batch_size
    );

    max_pool1d_kernel<<<blocks, threads>>>(
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