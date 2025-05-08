#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Shared-memory optimized max pooling 2D kernel
// Each block handles a tile of output elements for a single (batch, channel) slice.
// It first loads a corresponding patch (with halo for pooling windows) from the input into shared memory,
// then each thread performs pooling over its local window from the shared data.

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
    const int dilation) {

    // Determine the (batch, channel) that this block is processing
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;

    // Tile (block) dimensions for output
    // blockDim.x and blockDim.y indicate the number of output elements computed per block
    int tile_out_x = blockIdx.x * blockDim.x;
    int tile_out_y = blockIdx.y * blockDim.y;

    // Global output coordinate computed by each thread
    int out_x = tile_out_x + threadIdx.x;
    int out_y = tile_out_y + threadIdx.y;

    // Compute the corresponding top-left coordinate in the input for this tile
    // Each output element uses input starting at (out*stride - padding)
    int in_tile_origin_x = tile_out_x * stride - padding;
    int in_tile_origin_y = tile_out_y * stride - padding;

    // The shared memory tile must cover all input elements needed for the pooling windows
    // For a tile of output of size (tile_h, tile_w):
    // shared_width  = tile_w * stride + (kernel_size - 1) * dilation
    // shared_height = tile_h * stride + (kernel_size - 1) * dilation
    int tile_width  = blockDim.x;  // number of output columns in the tile
    int tile_height = blockDim.y;  // number of output rows in the tile
    int shared_width = tile_width * stride + (kernel_size - 1) * dilation;
    int shared_height = tile_height * stride + (kernel_size - 1) * dilation;
    int shared_size = shared_width * shared_height;

    // Declare dynamic shared memory
    extern __shared__ char smem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(smem);

    // Each thread loads one or more elements into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    // Compute the base pointer for the current (b, c) slice in the input
    int in_channel_offset = (b * channels + c) * input_height * input_width;

    // Load the shared memory tile from global memory
    for (int idx = tid; idx < shared_size; idx += total_threads) {
        int sh = idx / shared_width;  // row in shared memory tile
        int sw = idx % shared_width;  // column in shared memory tile
        int in_y = in_tile_origin_y + sh;
        int in_x = in_tile_origin_x + sw;
        if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
            shared_data[idx] = input[in_channel_offset + in_y * input_width + in_x];
        } else {
            shared_data[idx] = -std::numeric_limits<scalar_t>::infinity();
        }
    }

    __syncthreads();

    // Each thread now computes one output element using the data in shared memory
    if (out_y < output_height && out_x < output_width) {
        // The corresponding top-left coordinate in shared memory for this output element
        // is offset by (threadIdx.y * stride, threadIdx.x * stride)
        int shared_origin_y = threadIdx.y * stride;
        int shared_origin_x = threadIdx.x * stride;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int sy = shared_origin_y + kh * dilation;
                int sx = shared_origin_x + kw * dilation;
                int s_idx = sy * shared_width + sx;
                max_val = fmaxf(max_val, shared_data[s_idx]);
            }
        }

        // Write the computed max value to the output tensor
        int out_idx = (b * channels + c) * output_height * output_width + out_y * output_width + out_x;
        output[out_idx] = max_val;
    }
}


// Host function to launch the shared memory optimized max pooling kernel
torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // Calculate output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Set block (tile) dimensions. Here we use a 16x16 tile for output elements.
    const dim3 block(16, 16);
    const dim3 grid(
        (output_width  + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    // Compute shared memory size required per block
    int shared_width = block.x * stride + (kernel_size - 1) * dilation;
    int shared_height = block.y * stride + (kernel_size - 1) * dilation;
    size_t shared_mem_size = shared_width * shared_height * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_shared", ([&] {
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
