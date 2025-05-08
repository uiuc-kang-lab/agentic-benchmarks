#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses tiling to load input data into shared memory with a single synchronization
__global__ void tiled_avg_pool1d_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    // Determine batch and channel for this block
    int batch = blockIdx.z;
    int channel = blockIdx.y;

    // Each block processes a contiguous tile of output indices for one channel in one batch
    int o0 = blockIdx.x * blockDim.x;       // first output index processed by this block
    int o = o0 + threadIdx.x;                // each thread computes one output element
    if (o >= output_length) return;

    // Pointers for the current batch and channel
    const float* input_ptr = input + (batch * in_channels + channel) * input_length;
    float* output_ptr = output + (batch * in_channels + channel) * output_length;

    // Compute the global starting index for input data required for the tile
    // For the first output index o0, the corresponding input index is: o0*stride - padding
    int global_offset = o0 * stride - padding;

    // Calculate the tile size needed in shared memory.
    // For blockDim.x outputs, the loaded region length is: (blockDim.x * stride) + (kernel_size - stride)
    int tile_size = blockDim.x * stride + (kernel_size - stride);

    extern __shared__ float tile[];

    // Each thread cooperatively loads parts of the tile into shared memory.
    // Only one synchronization is used after loading to ensure all data is available for computation.
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        int global_idx = global_offset + i;
        if (global_idx < 0 || global_idx >= input_length) {
            tile[i] = 0.0f; // Treat out-of-bound indices as zero (padding)
        } else {
            tile[i] = input_ptr[global_idx];
        }
    }

    // Synchronize only once here to ensure the shared memory tile is fully loaded
    __syncthreads();

    // For the current output index, the starting position in the shared tile is given by:
    // local_index = (o - o0) * stride
    int local_index = (o - o0) * stride;
    float sum = 0.0f;
    
    // Use loop unrolling if possible to reduce loop overhead
    #pragma unroll
    for (int k = 0; k < kernel_size; k++) {
        sum += tile[local_index + k];
    }

    output_ptr[o] = sum / kernel_size;
}

// Host wrapper to launch the kernel
torch::Tensor tiled_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Configure block and grid dimensions. Each block handles a contiguous tile of outputs
    int threads = 256;  // Number of threads per block
    dim3 block(threads);
    dim3 grid((output_length + threads - 1) / threads, in_channels, batch_size);

    // Shared memory size needed per block
    int tile_size = threads * stride + (kernel_size - stride);
    size_t shared_mem_size = tile_size * sizeof(float);

    tiled_avg_pool1d_kernel<<<grid, block, shared_mem_size>>>(
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
    m.def("forward", &tiled_avg_pool1d_forward, "Tiled 1D Average Pooling forward (CUDA) with minimal synchronization");
}
