#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    const bool return_indices) {

    // Calculate base indices for this thread
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    const int c = blockIdx.y;
    const int b = blockIdx.z;

    if (i >= output_length || c >= num_channels || b >= batch_size) return;

    // Pre-calculate input base offset to reduce repeated calculations
    const int batch_offset = b * num_channels * input_length;
    const int channel_offset = c * input_length;
    const int input_base = batch_offset + channel_offset;
    
    // Calculate window boundaries
    const int window_start = i * stride - padding;
    const int window_end = window_start + (kernel_size - 1) * dilation;
    
    float max_val = -INFINITY;
    int max_idx = -1;

    // Bounds-checked loop with fewer conditionals
    const int start_k = (window_start < 0) ? (-window_start + dilation - 1) / dilation : 0;
    const int end_k = min(kernel_size, (input_length - window_start + dilation - 1) / dilation);

    #pragma unroll 4
    for (int k = start_k; k < end_k; ++k) {
        const int pos = window_start + k * dilation;
        const float val = input[input_base + pos];
        if (val > max_val) {
            max_val = val;
            max_idx = pos;
        }
    }

    // Calculate output offset once
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
    bool return_indices) {

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
                                 torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
    }

    const dim3 threads(256);
    const dim3 blocks((output_length + threads.x - 1) / threads.x, num_channels, batch_size);

    max_pool1d_kernel_optimized<<<blocks, threads>>>(
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
    m.def("forward", &forward, "MaxPool1D forward optimized (CUDA)");
}
