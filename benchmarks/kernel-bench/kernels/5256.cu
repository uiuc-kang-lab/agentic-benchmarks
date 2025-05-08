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

    // Shared memory for thread block's input data
    __shared__ float shared_input[32 * 4];  // For a 32x4 thread block

    // Combine position calculations to reduce register usage
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int i = blockIdx.x * blockDim.x + tid_x;
    const int c = blockIdx.y * blockDim.y + tid_y;
    const int b = blockIdx.z;

    if (b >= batch_size || c >= num_channels || i >= output_length) return;

    // Load parameters once and reuse
    const int input_start = i * kernel_params[1] - kernel_params[2];  // stride * i - padding
    const int input_offset = b * num_channels * kernel_params[4] + c * kernel_params[4];  // batch & channel offset
    
    // Initialize max tracking with register variables
    float max_val = -INFINITY;
    int max_idx = -1;

    // Unrolled loop for first elements to prime the pipeline
    #pragma unroll 4
    for (int k = 0; k < min(4, kernel_params[0]); ++k) {
        const int pos = input_start + k * kernel_params[3];  // Using dilation
        if (pos >= 0 && pos < kernel_params[4]) {
            const float val = input[input_offset + pos];
            if (val > max_val) {
                max_val = val;
                max_idx = pos;
            }
        }
    }

    // Main loop with fewer registers due to reuse
    for (int k = 4; k < kernel_params[0]; ++k) {
        const int pos = input_start + k * kernel_params[3];
        if (pos >= 0 && pos < kernel_params[4]) {
            const float val = input[input_offset + pos];
            max_val = val > max_val ? val : max_val;
            max_idx = val > max_val ? pos : max_idx;
        }
    }

    // Compute output index once
    const int out_idx = input_offset / kernel_params[4] * output_length + i;
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
