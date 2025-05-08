#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void max_pool1d_shared_memory_kernel(
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

    extern __shared__ float shared_input[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    int b = row / num_channels;
    int c = row % num_channels;

    if (i >= output_length) return;

    const int input_start = i * stride - padding;
    float max_val = -INFINITY;
    int max_idx = -1;

    const float* input_bc = input + (b * num_channels * input_length + c * input_length);

    for (int k = 0; k < kernel_size; ++k) {
        int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            shared_input[threadIdx.x] = input_bc[pos];
            __syncthreads();
            float val = shared_input[threadIdx.x];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
            __syncthreads();
        }
    }

    int out_index = b * (num_channels * output_length) + c * output_length + i;
    output[out_index] = max_val;
    if (return_indices)
        indices[out_index] = max_idx;
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {

    TORCH_CHECK(x.dim() == 3, "Input must be a 3D tensor.");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous.");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive.");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    const int threads_x = 256;
    dim3 threads(threads_x);
    dim3 blocks((output_length + threads_x - 1) / threads_x, batch_size * num_channels);

    size_t shared_mem_size = threads_x * sizeof(float);

    max_pool1d_shared_memory_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "MaxPool1D forward with optimized shared memory (CUDA)");
}
