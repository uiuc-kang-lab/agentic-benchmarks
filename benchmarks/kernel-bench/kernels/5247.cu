#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constant memory for kernel parameters
__constant__ int const_kernel_size;
__constant__ int const_stride;
__constant__ int const_padding;
__constant__ int const_dilation;

__global__ void max_pool1d_kernel_constant(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels,
    const int input_length,
    const int output_length,
    const bool return_indices) {

    const int b = blockIdx.z;
    const int c = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load constant parameters into registers
    const int kernel_size = const_kernel_size;
    const int stride = const_stride;
    const int padding = const_padding;
    const int dilation = const_dilation;

    if (b >= batch_size || c >= num_channels || i >= output_length) return;

    const int input_start = i * const_stride - const_padding;
    float max_val = -INFINITY;
    int max_idx = -1;

    for (int k = 0; k < const_kernel_size; ++k) {
        const int pos = input_start + k * const_dilation;
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

// Host function to launch the CUDA kernel
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

    // Copy parameters to constant memory
    cudaMemcpyToSymbol(const_kernel_size, &kernel_size, sizeof(int));
    cudaMemcpyToSymbol(const_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(const_padding, &padding, sizeof(int));
    cudaMemcpyToSymbol(const_dilation, &dilation, sizeof(int));

    const int block_size = 256;
    const dim3 threads(block_size);
    const dim3 blocks((output_length + block_size - 1) / block_size, num_channels, batch_size);

    max_pool1d_kernel_constant<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        num_channels,
        input_length,
        output_length,
        return_indices
    );

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with constant memory optimization (CUDA)");
}
