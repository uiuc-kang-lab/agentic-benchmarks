#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void max_pool1d_kernel_optimized(
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
    __shared__ float shared_input[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int output_idx = blockIdx.x * BLOCK_SIZE + tid;
    const int channel = blockIdx.y;
    const int batch = blockIdx.z;
    
    if (batch >= batch_size || channel >= num_channels) return;

    const int input_batch_offset = batch * num_channels * input_length;
    const int input_channel_offset = channel * input_length;
    const int output_offset = batch * num_channels * output_length + channel * output_length;

    if (output_idx < output_length) {
        const int input_start = output_idx * stride - padding;
        float max_val = -INFINITY;
        int max_idx = -1;

        #pragma unroll
        for (int k = 0; k < kernel_size; k++) {
            const int pos = input_start + k * dilation;
            if (pos >= 0 && pos < input_length) {
                const float val = input[input_batch_offset + input_channel_offset + pos];
                if (val > max_val) {
                    max_val = val;
                    max_idx = pos;
                }
            }
        }

        output[output_offset + output_idx] = max_val;
        if (return_indices) {
            indices[output_offset + output_idx] = max_idx;
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

    constexpr int BLOCK_SIZE = 128;
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks(
        (output_length + BLOCK_SIZE - 1) / BLOCK_SIZE,
        num_channels,
        batch_size
    );

    max_pool1d_kernel_optimized<BLOCK_SIZE><<<blocks, threads>>>(
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