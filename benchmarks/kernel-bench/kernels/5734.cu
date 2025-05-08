#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// This kernel uses shared memory to cache an input tile per block, reducing redundant global memory accesses.
// __syncthreads() is used only after the shared memory load to ensure consistency.

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
    // Each block computes a tile of output elements for one (b, c) slice.
    // Block dimensions correspond to output tile dimensions.
    const int out_tile_w = blockDim.x;
    const int out_tile_h = blockDim.y;
    
    // Compute the global output coordinates for this thread
    int out_x = blockIdx.x * out_tile_w + threadIdx.x;
    int out_y = blockIdx.y * out_tile_h + threadIdx.y;

    // Each block processes one channel for one batch element
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;

    if (out_x >= output_width || out_y >= output_height || b >= batch_size || c >= channels)
        return;
        
    // Compute the corresponding top-left coordinate in the input for this output
    int in_x = out_x * stride - padding;
    int in_y = out_y * stride - padding;

    // Determine the input tile to load into shared memory for this block
    // The tile covers all pooling windows for outputs in this block.
    int tile_origin_x = blockIdx.x * out_tile_w * stride - padding;
    int tile_origin_y = blockIdx.y * out_tile_h * stride - padding;
    int tile_width = out_tile_w * stride + (kernel_size - 1) * dilation;
    int tile_height = out_tile_h * stride + (kernel_size - 1) * dilation;
    int tile_size = tile_width * tile_height;

    extern __shared__ char smem_raw[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(smem_raw);

    // Load the required tile from global memory into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;

    // Pointer to the input channel
    const scalar_t* in_ptr = input + (b * channels + c) * input_height * input_width;

    for (int i = tid; i < tile_size; i += block_threads) {
        int ty = i / tile_width;
        int tx = i % tile_width;
        int gx = tile_origin_x + tx;
        int gy = tile_origin_y + ty;
        if (gx >= 0 && gx < input_width && gy >= 0 && gy < input_height) {
            shmem[i] = in_ptr[gy * input_width + gx];
        } else {
            shmem[i] = -std::numeric_limits<scalar_t>::infinity();
        }
    }
    __syncthreads();  // Synchronize to ensure shared memory load is complete

    // Each thread computes its pooling output using the shared memory tile.
    // The pooling window in shared memory starts at (local_base_x, local_base_y):
    int local_base_x = in_x - tile_origin_x;
    int local_base_y = in_y - tile_origin_y;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    if (kernel_size == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            int ly = local_base_y + kh * dilation;
            #pragma unroll
            for (int kw = 0; kw < 2; kw++) {
                int lx = local_base_x + kw * dilation;
                max_val = max(max_val, shmem[ly * tile_width + lx]);
            }
        }
    } else if (kernel_size == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            int ly = local_base_y + kh * dilation;
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int lx = local_base_x + kw * dilation;
                max_val = max(max_val, shmem[ly * tile_width + lx]);
            }
        }
    } else {
        for (int kh = 0; kh < kernel_size; kh++) {
            int ly = local_base_y + kh * dilation;
            for (int kw = 0; kw < kernel_size; kw++) {
                int lx = local_base_x + kw * dilation;
                max_val = max(max_val, shmem[ly * tile_width + lx]);
            }
        }
    }
    
    // Write the computed max value to the output tensor
    int out_index = ((b * channels + c) * output_height + out_y) * output_width + out_x;
    output[out_index] = max_val;
}


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

    // Define block and grid dimensions based on output dimensions
    const dim3 threads(32, 8);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    // Calculate the shared memory size required per block
    int tile_width = threads.x * stride + (kernel_size - 1) * dilation;
    int tile_height = threads.y * stride + (kernel_size - 1) * dilation;
    size_t shared_mem_size = tile_width * tile_height * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_shared_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}
