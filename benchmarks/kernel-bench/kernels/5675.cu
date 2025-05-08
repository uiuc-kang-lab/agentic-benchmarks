#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Shared-memory based max pooling 2D kernel ensuring coalesced global memory accesses
template <typename scalar_t>
__global__ void max_pool2d_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    // Each block handles one (batch, channel) slice
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;

    // Global output coordinate computed per thread
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Determine shared memory tile dimensions
    // The tile covers the input region needed by the entire output block
    const int sm_w = blockDim.x * stride + (kernel_size - 1) * dilation;
    const int sm_h = blockDim.y * stride + (kernel_size - 1) * dilation;

    // Compute the top-left corner in the input corresponding to the block
    const int in_tile_x = blockIdx.x * blockDim.x * stride - padding;
    const int in_tile_y = blockIdx.y * blockDim.y * stride - padding;

    // Allocate shared memory (declared as bytes, then cast to scalar_t pointer)
    extern __shared__ char shared_mem[];
    scalar_t* tile = reinterpret_cast<scalar_t*>(shared_mem);

    const int tile_size = sm_w * sm_h;
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;

    // Cooperative loading of the input tile into shared memory
    for (int i = thread_id; i < tile_size; i += block_threads) {
        const int ty = i / sm_w;
        const int tx = i % sm_w;
        const int in_x = in_tile_x + tx;
        const int in_y = in_tile_y + ty;
        scalar_t value;
        if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
            const int input_idx = b * (channels * input_height * input_width) +
                                  c * (input_height * input_width) +
                                  in_y * input_width + in_x;
            value = input[input_idx];
        } else {
            value = -std::numeric_limits<scalar_t>::infinity();
        }
        tile[i] = value;
    }
    __syncthreads();

    // Only compute output if within valid output range
    if (out_x < output_width && out_y < output_height) {
        // The pooling window for this output element starts at this offset in shared memory
        const int tile_x = threadIdx.x * stride;
        const int tile_y = threadIdx.y * stride;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        // Iterate over the pooling window
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                const int sx = tile_x + j * dilation;
                const int sy = tile_y + i * dilation;
                const int index = sy * sm_w + sx;
                const scalar_t candidate = tile[index];
                if (candidate > max_val) {
                    max_val = candidate;
                }
            }
        }

        const int output_idx = b * (channels * output_height * output_width) +
                               c * (output_height * output_width) +
                               out_y * output_width + out_x;
        output[output_idx] = max_val;
    }
}

// Host function wrapper
torch::Tensor max_pool2d_shared_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Use a 16x16 thread block to improve memory coalescing on global accesses
    const dim3 block(16, 16);
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    // Calculate shared memory size in bytes
    const int sm_w = block.x * stride + (kernel_size - 1) * dilation;
    const int sm_h = block.y * stride + (kernel_size - 1) * dilation;
    size_t shared_mem_size = sm_w * sm_h * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_shared_cuda_forward", ([&] {
        max_pool2d_shared_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_shared_cuda_forward, "Max Pool 2D forward with shared memory coalescing (CUDA)");
}
