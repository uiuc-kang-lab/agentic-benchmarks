/*
Combined 3D average pooling kernel: 
This version integrates the straightforward indexing approach from Kernel 1 with the shared memory tiling technique from Kernel 2. 
It loads a shared input tile once per block and then each thread computes an output element by summing over a pooling window from the shared memory tile. 
This approach improves global memory reuse and reduces redundant loads across overlapping pooling windows.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tiling sizes for the output tile processed by each block
#define TILE_D 4
#define TILE_H 4
#define TILE_W 4

// Combined CUDA kernel using shared memory tiling and loop unrolling directives
__global__ void avg_pool3d_forward_combined_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    // Map grid.x to batch and channel (n, c) pair
    int bc = blockIdx.x; // ranges from 0 to batch_size*channels - 1
    int n = bc / channels;
    int c = bc % channels;

    // Define output tile sizes (for depth, height, width) from macros
    const int tile_d = TILE_D;
    const int tile_h = TILE_H;
    const int tile_w = TILE_W;

    // Compute number of tiles along each spatial dimension
    int num_tiles_d = (out_d + tile_d - 1) / tile_d;
    int num_tiles_h = (out_h + tile_h - 1) / tile_h;
    int num_tiles_w = (out_w + tile_w - 1) / tile_w;

    // Determine the tile index for the output
    int d_tile_idx = blockIdx.y;  // tile index along depth
    int d_out_start = d_tile_idx * tile_d;

    // Decompose grid.z into tile indices for height and width
    int tile_idx = blockIdx.z;    
    int h_tile_idx = tile_idx / num_tiles_w;  
    int w_tile_idx = tile_idx % num_tiles_w;
    int h_out_start = h_tile_idx * tile_h;
    int w_out_start = w_tile_idx * tile_w;

    // Compute actual tile sizes (tile may be smaller at boundaries)
    int actual_tile_d = (d_out_start + tile_d > out_d) ? (out_d - d_out_start) : tile_d;
    int actual_tile_h = (h_out_start + tile_h > out_h) ? (out_h - h_out_start) : tile_h;
    int actual_tile_w = (w_out_start + tile_w > out_w) ? (out_w - w_out_start) : tile_w;

    // Calculate the dimensions of the shared memory region needed.
    // Each output element’s pooling window starts at: output_index*stride - padding.
    // For a tile, the shared memory region covers all pooling windows:
    int shared_d = (actual_tile_d - 1) * stride + kernel_size;
    int shared_h = (actual_tile_h - 1) * stride + kernel_size;
    int shared_w = (actual_tile_w - 1) * stride + kernel_size;

    // Compute the corresponding starting indices in the input tensor for the tile
    int d_in_start = d_out_start * stride - padding;
    int h_in_start = h_out_start * stride - padding;
    int w_in_start = w_out_start * stride - padding;

    // Allocate shared memory (dynamically sized) as a 1D array representing a 3D block
    extern __shared__ float shmem[];
    int shared_tile_size = shared_d * shared_h * shared_w;

    // Each thread in the block cooperatively loads portions of the shared memory tile
    int tid = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y * blockDim.z;
    for (int idx = tid; idx < shared_tile_size; idx += block_threads) {
        // Map linear index to 3D coordinates in the shared memory tile
        int s_d = idx / (shared_h * shared_w);
        int rem = idx % (shared_h * shared_w);
        int s_h = rem / shared_w;
        int s_w = rem % shared_w;

        // Compute corresponding global input indices
        int in_d_idx = d_in_start + s_d;
        int in_h_idx = h_in_start + s_h;
        int in_w_idx = w_in_start + s_w;

        float val = 0.0f;
        if (in_d_idx >= 0 && in_d_idx < in_d) {
            in_h_idx = min(in_h - 1, in_h_start + s_h);
            in_w_idx = min(in_w - 1, in_w_start + s_w);
            int input_index = (((n * channels + c) * in_d + in_d_idx) * in_h + in_h_idx) * in_w + in_w_idx;
            val = input[input_index];
        }
        shmem[idx] = val;
    }

    __syncthreads();

    // Each thread computes one output element within the tile if its thread indices are in range
    int t_w = threadIdx.x; // local coordinate in tile width
    int t_h = threadIdx.y; // local coordinate in tile height
    int t_d = threadIdx.z; // local coordinate in tile depth

    if (t_w < actual_tile_w && t_h < actual_tile_h && t_d < actual_tile_d) {
        int d_out = d_out_start + t_d;
        int h_out = h_out_start + t_h;
        int w_out = w_out_start + t_w;

        // Determine the starting offset for the pooling window in shared memory
        // Since d_in_start corresponds to d_out_start*stride - padding, the local start is:
        int s_d_start = t_d * stride;
        int s_h_start = t_h * stride;
        int s_w_start = t_w * stride;
        
        float sum = 0.0f;
        // Compute pooling sum from the shared memory tile
        // Using conventional loops. For small kernel sizes the compiler may unroll these loops.
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int s_d = s_d_start + kd;
                    int s_h = s_h_start + kh;
                    int s_w = s_w_start + kw;
                    int shmem_idx = (s_d * shared_h * shared_w) + (s_h * shared_w) + s_w;
                    sum += shmem[shmem_idx];
                }
            }
        }

        int pool_volume = kernel_size * kernel_size * kernel_size;
        float avg = sum / static_cast<float>(pool_volume);

        // Compute the global output index
        int out_index = ((((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out);
        output[out_index] = avg;
    }
}

// Host function: set up grid/block dimensions and launch the kernel
at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional (N, C, D, H, W)");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    // Compute output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    // Determine tiling parameters (see kernel: grid.x covers (n, c), grid.y covers depth, grid.z covers height and width tiles)
    int num_tiles_d = (out_d + TILE_D - 1) / TILE_D;
    int num_tiles_h = (out_h + TILE_H - 1) / TILE_H;
    int num_tiles_w = (out_w + TILE_W - 1) / TILE_W;
    
    // Grid dimensions
    // grid.x -> batch_size * channels
    // grid.y -> number of tiles along output depth
    // grid.z -> combined index for tiles along height and width
    dim3 grid(batch_size * channels, num_tiles_d, num_tiles_h * num_tiles_w);
    // Each block handles a tile of size (TILE_W, TILE_H, TILE_D)
    dim3 block(TILE_W, TILE_H, TILE_D);

    // Shared memory size per block
    int shared_d = ((TILE_D - 1) * stride + kernel_size);
    int shared_h = ((TILE_H - 1) * stride + kernel_size);
    int shared_w = ((TILE_W - 1) * stride + kernel_size);
    size_t shared_mem_size = shared_d * shared_h * shared_w * sizeof(float);

    avg_pool3d_forward_combined_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (Combined CUDA)");
}
