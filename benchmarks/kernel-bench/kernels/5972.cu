#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using shared memory tiling for 1D average pooling
__global__ void sharedmem_avg_pool1d_kernel(
    const float * __restrict__ input,
    float * __restrict__ output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    int tid = threadIdx.x;
    int o = blockIdx.x * blockDim.x + tid;  // Global output index
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    // Compute the starting global input index for the tile
    int tile_start_global = blockIdx.x * blockDim.x * stride - padding;
    // Tile size covers all input elements needed for output elements in this block
    int tile_size = kernel_size + (blockDim.x - 1) * stride;

    extern __shared__ float smem[];

    // Each thread cooperatively loads the shared memory tile
    for (int i = tid; i < tile_size; i += blockDim.x) {
        int global_idx = tile_start_global + i;
        if (global_idx >= 0 && global_idx < input_length) {
            smem[i] = input[batch * in_channels * input_length + channel * input_length + global_idx];
        } else {
            smem[i] = 0.0f;  // Zero-padding for out-of-bound indices
        }
    }

    // Synchronize threads after loading shared memory
    __syncthreads();

    if (o < output_length) {
        // Compute the offset in shared memory corresponding to the start of the pooling window
        int smem_offset = tid * stride;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            sum += smem[smem_offset + k];
        }
        // Write the averaged result to global memory
        output[batch * in_channels * output_length + channel * output_length + o] = sum / kernel_size;
    }
}

// Host function to launch the kernel
torch::Tensor sharedmem_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    int threads = 256;
    int grid_x = (output_length + threads - 1) / threads;  // Number of blocks in the output dimension
    dim3 grid(grid_x, in_channels, batch_size);
    dim3 block(threads);

    // Calculate shared memory size: tile covers kernel_size + (threads - 1) * stride floats
    int tile_size = kernel_size + (threads - 1) * stride;
    size_t shmem_size = tile_size * sizeof(float);

    sharedmem_avg_pool1d_kernel<<<grid, block, shmem_size>>>(
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
    m.def("forward", &sharedmem_avg_pool1d_forward, "1D Average Pooling forward (CUDA) with shared memory tiling");
}
