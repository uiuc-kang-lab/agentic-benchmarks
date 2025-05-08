#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile sizes for output (you can tune these values)
#define TILE_D 4
#define TILE_H 4
#define TILE_W 4

// CUDA kernel implementing 3D Average Pooling using shared memory tiling.
// Each block processes a tile of output for a given (n, c) slice.
// Threads cooperatively load the corresponding patch of input into shared memory.
// Only one __syncthreads() is used after the shared memory load to ensure consistency.

__global__ void avg_pool3d_forward_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // We pack the (n, c) indices in grid.x
    int bc = blockIdx.x;  // range: 0 .. batch_size*channels - 1
    int n = bc / channels;
    int c = bc % channels;

    // Determine tiling for the output spatial dims using grid.y and grid.z.
    // grid.y indexes the tile along the output depth dimension.
    // grid.z is a combined index for the tile in the height and width dims.
    int tile_d = TILE_D;  
    int tile_h = TILE_H;
    int tile_w = TILE_W;

    int num_tiles_d = (out_d + tile_d - 1) / tile_d;
    int num_tiles_h = (out_h + tile_h - 1) / tile_h;
    int num_tiles_w = (out_w + tile_w - 1) / tile_w;

    // Identify the tile for depth from grid.y
    int d_tile_idx = blockIdx.y; // in [0, num_tiles_d)
    int d_out_start = d_tile_idx * tile_d;

    // Decompose grid.z into tile indices for height and width.
    int tile_idx = blockIdx.z;
    int h_tile_idx = tile_idx / num_tiles_w;  // integer division
    int w_tile_idx = tile_idx % num_tiles_w;
    int h_out_start = h_tile_idx * tile_h;
    int w_out_start = w_tile_idx * tile_w;

    // Compute the actual number of output elements in this tile (may be smaller at boundaries)
    int actual_tile_d = (d_out_start + tile_d > out_d) ? (out_d - d_out_start) : tile_d;
    int actual_tile_h = (h_out_start + tile_h > out_h) ? (out_h - h_out_start) : tile_h;
    int actual_tile_w = (w_out_start + tile_w > out_w) ? (out_w - w_out_start) : tile_w;

    // Determine the dimensions of the shared memory region.
    // For each output element, the pooling window starts at (d*out_stride - padding).
    // The union of the pooling windows for the tile spans:
    // depth: from d_out_start*stride - padding to (d_out_start + actual_tile_d - 1)*stride - padding + kernel_size
    int shared_d = ((actual_tile_d - 1) * stride + kernel_size);
    int shared_h = ((actual_tile_h - 1) * stride + kernel_size);
    int shared_w = ((actual_tile_w - 1) * stride + kernel_size);

    // Compute the starting global indices in input corresponding to the top-left-front corner of the tile.
    int d_in_start = d_out_start * stride - padding;
    int h_in_start = h_out_start * stride - padding;
    int w_in_start = w_out_start * stride - padding;

    // Allocate shared memory (dynamically sized) as a 1D array representing a 3D block
    extern __shared__ float shmem[];
    int shared_tile_size = shared_d * shared_h * shared_w;

    // Each thread in the block loads part of the shared memory tile.
    int tid = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y * blockDim.z;
    for (int idx = tid; idx < shared_tile_size; idx += block_threads) {
        // Map linear index to 3D coordinates in shared memory
        int s_d = idx / (shared_h * shared_w);
        int rem = idx % (shared_h * shared_w);
        int s_h = rem / shared_w;
        int s_w = rem % shared_w;

        // Compute corresponding global input indices
        int in_d_idx = d_in_start + s_d;
        int in_h_idx = h_in_start + s_h;
        int in_w_idx = w_in_start + s_w;

        float val = 0.0f;
        if (in_d_idx >= 0 && in_d_idx < in_d &&
            in_h_idx >= 0 && in_h_idx < in_h &&
            in_w_idx >= 0 && in_w_idx < in_w) {
            // Compute linear index for input: index = (((n * channels + c) * in_d + in_d_idx) * in_h + in_h_idx) * in_w + in_w_idx
            int input_index = (((n * channels + c) * in_d + in_d_idx) * in_h + in_h_idx) * in_w + in_w_idx;
            val = input[input_index];
        }
        shmem[idx] = val;
    }

    // Synchronize threads after shared memory load
    __syncthreads();

    // Now, each thread in the block computes one output element from the tile.
    // We assume the block is launched with dimensions (tile_w, tile_h, tile_d).
    int t_w = threadIdx.x;  // should be in [0, actual_tile_w)
    int t_h = threadIdx.y;  // in [0, actual_tile_h)
    int t_d = threadIdx.z;  // in [0, actual_tile_d)

    if (t_w < actual_tile_w && t_h < actual_tile_h && t_d < actual_tile_d) {
        int d_out = d_out_start + t_d;
        int h_out = h_out_start + t_h;
        int w_out = w_out_start + t_w;

        // Compute the starting point of the pooling window in global input
        int in_d_pool = d_out * stride - padding;
        int in_h_pool = h_out * stride - padding;
        int in_w_pool = w_out * stride - padding;
        
        // Compute the corresponding starting indices in shared memory
        int s_d_start = in_d_pool - d_in_start;
        int s_h_start = in_h_pool - h_in_start;
        int s_w_start = in_w_pool - w_in_start;

        float sum = 0.0f;
        // Sum over the pooling window from shared memory
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int s_d_idx = s_d_start + kd;
                    int s_h_idx = s_h_start + kh;
                    int s_w_idx = s_w_start + kw;
                    // Check bounds within shared memory tile
                    if (s_d_idx >= 0 && s_d_idx < shared_d &&
                        s_h_idx >= 0 && s_h_idx < shared_h &&
                        s_w_idx >= 0 && s_w_idx < shared_w) {
                        int shmem_idx = (s_d_idx * shared_h * shared_w) + (s_h_idx * shared_w) + s_w_idx;
                        sum += shmem[shmem_idx];
                    } else {
                        sum += 0.0f;  // out-of-bound values (padding) as 0
                    }
                }
            }
        }
        
        float avg = sum / (kernel_size * kernel_size * kernel_size);

        // Write the result to the output tensor.
        // Output index: ((((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out)
        int out_index = (((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out;
        output[out_index] = avg;
    }
}

// Host function: set up grid/block dims and launch the kernel
at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    // Check input tensor
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional (N, C, D, H, W)");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);

    // Compute output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    // Determine tiling parameters for the output
    int tile_d = TILE_D;
    int tile_h = TILE_H;
    int tile_w = TILE_W;

    int num_tiles_d = (out_d + tile_d - 1) / tile_d;
    int num_tiles_h = (out_h + tile_h - 1) / tile_h;
    int num_tiles_w = (out_w + tile_w - 1) / tile_w;

    // Grid dimensions: 
    // grid.x -> batch_size * channels
    // grid.y -> tiles along depth
    // grid.z -> combined tile index for height and width (row-major: tile_h_index * num_tiles_w + tile_w_index)
    dim3 grid(batch_size * channels, num_tiles_d, num_tiles_h * num_tiles_w);

    // Block dimensions: each block covers one tile with dimensions (tile_w, tile_h, tile_d)
    dim3 block(tile_w, tile_h, tile_d);

    // Shared memory size required per block
    int shared_d = ((TILE_D - 1) * stride + kernel_size);
    int shared_h = ((TILE_H - 1) * stride + kernel_size);
    int shared_w = ((TILE_W - 1) * stride + kernel_size);
    size_t shared_mem_size = shared_d * shared_h * shared_w * sizeof(float);

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    avg_pool3d_forward_tiled_kernel<<<grid, block, shared_mem_size>>>(
        input_ptr, output_ptr,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward tiled (CUDA)");
}
