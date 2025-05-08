#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// This kernel uses shared memory to load a tile of the input and then computes the max pooling
// result using the data in shared memory. Only one __syncthreads() is used after loading
// the shared memory tile to ensure consistency, avoiding excessive synchronizations.

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
    // Each block processes one (batch, channel) pair
    int bc = blockIdx.z; 
    int b = bc / channels;
    int c = bc % channels;

    // Determine the starting coordinate of the output tile for this block
    int out_tile_row = blockIdx.y * blockDim.y;
    int out_tile_col = blockIdx.x * blockDim.x;

    // Corresponding starting coordinate in the input
    int in_tile_row = out_tile_row * stride - padding;
    int in_tile_col = out_tile_col * stride - padding;

    // Dimensions of the shared memory tile
    int tile_rows = blockDim.y * stride + (kernel_size - 1) * dilation;
    int tile_cols = blockDim.x * stride + (kernel_size - 1) * dilation;

    // Allocate shared memory dynamically
    extern __shared__ char smem[];
    scalar_t* s_tile = reinterpret_cast<scalar_t*>(smem);

    int tile_size = tile_rows * tile_cols;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;

    // Base pointer for this (b, c) in the input
    int input_channel_offset = b * channels * input_height * input_width + c * input_height * input_width;

    // Load the shared memory tile cooperatively
    for (int idx = thread_id; idx < tile_size; idx += block_threads) {
        int tile_r = idx / tile_cols;
        int tile_c = idx % tile_cols;
        int global_r = in_tile_row + tile_r;
        int global_c = in_tile_col + tile_c;
        if (global_r >= 0 && global_r < input_height && global_c >= 0 && global_c < input_width) {
            s_tile[idx] = input[input_channel_offset + global_r * input_width + global_c];
        } else {
            s_tile[idx] = -std::numeric_limits<scalar_t>::infinity();
        }
    }
    __syncthreads(); // Ensure shared memory is fully loaded

    // Each thread computes one output element
    int out_r = out_tile_row + threadIdx.y;
    int out_c = out_tile_col + threadIdx.x;
    if (out_r < output_height && out_c < output_width) {
        // In shared memory, the pooling window for this output starts at
        // (threadIdx.y * stride, threadIdx.x * stride)
        int local_r = threadIdx.y * stride;
        int local_c = threadIdx.x * stride;
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        // Loop over the pooling kernel window
        for (int kh = 0; kh < kernel_size; ++kh) {
            int sr = local_r + kh * dilation;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int sc = local_c + kw * dilation;
                int sh_index = sr * tile_cols + sc;
                scalar_t val = s_tile[sh_index];
                max_val = (val > max_val) ? val : max_val;
            }
        }
        
        // Write the result to the global output
        int out_index = b * channels * output_height * output_width +
                        c * output_height * output_width +
                        out_r * output_width + out_c;
        output[out_index] = max_val;
    }
}

// Host function to launch the shared memory optimized max pooling kernel
torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Choose a reasonable block size; here we use 8x8 threads per block
    const dim3 block(8, 8);
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    // Calculate shared memory size per block
    int tile_rows = block.y * stride + (kernel_size - 1) * dilation;
    int tile_cols = block.x * stride + (kernel_size - 1) * dilation;
    size_t shared_mem_size = tile_rows * tile_cols * sizeof(float);  // adjust if using different types

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with shared memory optimization (CUDA)");
}
