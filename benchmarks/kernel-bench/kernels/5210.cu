#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void coalesced_max_pool1d_kernel(
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
    const int tidx = threadIdx.x;
    const int lane_id = tidx & 31;  // Thread index within warp
    const int warp_id = tidx >> 5;  // Warp index within block
    
    const int global_idx = (blockIdx.x * (blockDim.x >> 5) + warp_id) * 32 + lane_id;
    
    if (global_idx < output_length * num_channels * batch_size) {
        const int o = global_idx % output_length;
        const int c = (global_idx / output_length) % num_channels;
        const int b = global_idx / (output_length * num_channels);

        const int input_start = o * stride - padding;
        const int base_idx = b * (num_channels * input_length) + c * input_length;
        
        float max_val = -INFINITY;
        int max_idx = -1;

        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            const int pos = input_start + k * dilation;
            if (pos >= 0 && pos < input_length) {
                const float val = __ldg(&input[base_idx + pos]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = pos;
                }
            }
        }

        output[global_idx] = max_val;
        if (return_indices) {
            indices[global_idx] = max_idx;
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

    const int total_elements = batch_size * num_channels * output_length;
    const int threads_per_block = 128;
    const int num_warps_needed = (total_elements + 31) / 32;
    const int warps_per_block = threads_per_block / 32;
    const int num_blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;

    coalesced_max_pool1d_kernel<<<num_blocks, threads_per_block>>>(
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
    m.def("forward", &forward, "MaxPool1D forward with coalesced memory access (CUDA)");
}