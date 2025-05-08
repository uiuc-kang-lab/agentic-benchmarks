#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel leveraging shared memory for 1D average pooling
__global__ void shared_memory_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int input_length,
    const int output_length,
    const int batch_size,
    const int in_channels) {

    // Each block processes a contiguous chunk of output elements for a single (batch, channel)
    // Determine the current batch and channel from grid dimensions
    const int channel = blockIdx.y;
    const int batch = blockIdx.z;

    // Compute the first output index handled by this block
    const int o_first = blockIdx.x * blockDim.x;
    // Global output index for this thread
    const int o = o_first + threadIdx.x;

    // Pointer to the start of the current input row
    const int input_offset = batch * in_channels * input_length + channel * input_length;
    
    // Compute the start index in the input corresponding to the block's first output element
    // This may be negative if padding is large
    int t_start = o_first * stride - padding;
    // The total number of input elements needed for the block's outputs
    int tile_size = (blockDim.x - 1) * stride + kernel_size;

    extern __shared__ float s_data[];

    // Load the required input tile into shared memory
    // Each thread loads one or more elements in a strided loop
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        int global_index = t_start + i;
        if (global_index >= 0 && global_index < input_length)
            s_data[i] = input[input_offset + global_index];
        else
            s_data[i] = 0.0f;
    }
    __syncthreads();

    // Only compute if the output index is within bounds
    if (o < output_length) {
        // Map the output's required input start index into shared memory index
        int s_index = (o * stride - padding) - t_start;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            sum += s_data[s_index + k];
        }
        output[batch * (in_channels * output_length) + channel * output_length + o] = sum / kernel_size;
    }
}

// Host function to launch the shared memory kernel
torch::Tensor shared_memory_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid pooling parameters");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Set a fixed block size for output elements (tune as needed)
    const int threads = 256;
    const int blocks_x = (output_length + threads - 1) / threads;
    dim3 blocks(blocks_x, in_channels, batch_size);
    dim3 threadsDim(threads);

    // Compute the shared memory size required per block
    int tile_size = (threads - 1) * stride + kernel_size;
    size_t shared_mem_size = tile_size * sizeof(float);

    shared_memory_avg_pool1d_kernel<<<blocks, threadsDim, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        batch_size,
        in_channels
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_avg_pool1d_forward, "Shared Memory 1D Average Pooling forward (CUDA)");
}
