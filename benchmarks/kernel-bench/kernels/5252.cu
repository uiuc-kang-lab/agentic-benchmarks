#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use constant memory to cache kernel parameters for all threads
__constant__ int kernel_params[5]; // {kernel_size, stride, padding, dilation, input_length}

// Optimized kernel: uses a 2D thread block (over output length and channels) and constant memory
__global__ void max_pool1d_kernel_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels,
    const int output_length,
    const bool return_indices) {

    // Determine our position in the output tensor
    int i = blockIdx.x * blockDim.x + threadIdx.x; // output position along the length
    int c = blockIdx.y * blockDim.y + threadIdx.y; // channel index
    int b = blockIdx.z; // batch index

    if (b >= batch_size || c >= num_channels || i >= output_length) return;

    // Load parameters from constant memory
    int kernel_size = kernel_params[0];
    int stride      = kernel_params[1];
    int padding     = kernel_params[2];
    int dilation    = kernel_params[3];
    int input_length= kernel_params[4];

    // Compute start index into the input
    int input_start = i * stride - padding;
    float max_val = -INFINITY;
    int max_idx = -1;

    // Iterate over the pooling window
    for (int k = 0; k < kernel_size; ++k) {
        int pos = input_start + k * dilation;
        if (pos >= 0 && pos < input_length) {
            float val = input[b * num_channels * input_length + c * input_length + pos];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }

    // Write the results
    int out_idx = b * num_channels * output_length + c * output_length + i;
    output[out_idx] = max_val;
    if (return_indices) {
        indices[out_idx] = max_idx;
    }
}

// Host function that sets up the GPU kernel launch
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

    const int batch_size   = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    // Calculate the output length
    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length},
                                 torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
    }

    // Copy parameters to constant memory so all threads can efficiently access them
    int host_params[5] = {
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(dilation),
        static_cast<int>(input_length)};
    cudaMemcpyToSymbol(kernel_params, host_params, sizeof(host_params));

    // Define a 2D block over output length and channel dimension, with batch as the z-dimension
    const dim3 threads(32, 4);
    const dim3 blocks(
        (output_length + threads.x - 1) / threads.x,
        (num_channels + threads.y - 1) / threads.y,
        batch_size);

    // Launch the optimized kernel
    max_pool1d_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        num_channels,
        output_length,
        return_indices);

    // Optionally, one might add cudaDeviceSynchronize() to catch errors in debug mode
    // cudaDeviceSynchronize();

    // Return concatenated tensor if indices were requested
    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized MaxPool1D forward (CUDA)");
}
