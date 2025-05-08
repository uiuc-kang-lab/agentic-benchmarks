#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for output per block
#define TILE_W 8
#define TILE_H 8
#define TILE_D 4

// This kernel uses shared memory to cache a tile of the input corresponding to a tile of outputs
// Only one __syncthreads() is used after loading shared memory, ensuring minimal synchronization overhead.

__global__ void avg_pool3d_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size,
    int stride,
    int padding) {

    // Determine how many depth tiles per (n, c) slice
    int d_tiles = (out_d + TILE_D - 1) / TILE_D;
    
    // Decode blockIdx.z to get the (n, c) slice and depth tile index
    int slice_idx = blockIdx.z / d_tiles;  // slice index over batch*channels
    int tile_d_index = blockIdx.z % d_tiles; // which tile along depth within this slice
    
    int n = slice_idx / channels;
    int c = slice_idx % channels;

    // Determine the starting coordinates for this output tile
    int out_d_start = tile_d_index * TILE_D;
    int out_h_start = blockIdx.y * TILE_H;
    int out_w_start = blockIdx.x * TILE_W;

    // Compute corresponding starting coordinates in the input
    int in_d_start = out_d_start * stride - padding;
    int in_h_start = out_h_start * stride - padding;
    int in_w_start = out_w_start * stride - padding;

    // Effective dimensions of the input tile loaded into shared memory
    int tile_d_eff = (TILE_D - 1) * stride + kernel_size;
    int tile_h_eff = (TILE_H - 1) * stride + kernel_size;
    int tile_w_eff = (TILE_W - 1) * stride + kernel_size;
    int shared_tile_size = tile_d_eff * tile_h_eff * tile_w_eff;

    extern __shared__ float shared_mem[];

    // Total number of threads in the block
    int block_threads = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;

    // Each thread loads one or more elements of the shared memory tile
    for (int i = tid; i < shared_tile_size; i += block_threads) {
        // Decode linear index i into 3D coordinates (sd, sh, sw) in the shared tile
        int sd = i / (tile_h_eff * tile_w_eff);
        int rem = i % (tile_h_eff * tile_w_eff);
        int sh = rem / tile_w_eff;
        int sw = rem % tile_w_eff;

        // Map to global input coordinates
        int global_d = in_d_start + sd;
        int global_h = in_h_start + sh;
        int global_w = in_w_start + sw;

        float val = 0.0f;
        if (global_d >= 0 && global_d < in_d && global_h >= 0 && global_h < in_h && global_w >= 0 && global_w < in_w) {
            int input_index = (((n * channels + c) * in_d + global_d) * in_h + global_h) * in_w + global_w;
            val = input[input_index];
        }
        shared_mem[i] = val;
    }

    // Synchronize to ensure the shared memory tile is fully loaded
    __syncthreads();

    // Each thread computes one output element within the tile
    int tx = threadIdx.x; // local output w index
    int ty = threadIdx.y; // local output h index
    int tz = threadIdx.z; // local output d index

    int out_w_idx = out_w_start + tx;
    int out_h_idx = out_h_start + ty;
    int out_d_idx = out_d_start + tz;

    // Check output bounds
    if (out_w_idx < out_w && out_h_idx < out_h && out_d_idx < out_d) {
        // In the shared memory tile, the pooling window for this output starts at:
        int sm_w_start = tx * stride;
        int sm_h_start = ty * stride;
        int sm_d_start = tz * stride;

        float sum = 0.0f;
        // Sum over the pooling window of size kernel_size^3 in shared memory
        for (int kd = 0; kd < kernel_size; kd++) {
            int sm_d = sm_d_start + kd;
            for (int kh = 0; kh < kernel_size; kh++) {
                int sm_h = sm_h_start + kh;
                for (int kw = 0; kw < kernel_size; kw++) {
                    int sm_w = sm_w_start + kw;
                    int sm_index = (sm_d * tile_h_eff + sm_h) * tile_w_eff + sm_w;
                    sum += shared_mem[sm_index];
                }
            }
        }
        
        // Compute the average (always dividing by the full kernel volume, count_include_pad = true)
        float avg = sum / (kernel_size * kernel_size * kernel_size);
        
        // Write the result to the output tensor; output tensor is 5D: [batch, channels, out_d, out_h, out_w]
        int output_index = (((n * channels + c) * out_d + out_d_idx) * out_h + out_h_idx) * out_w + out_w_idx;
        output[output_index] = avg;
    }
    // No extra synchronization is needed after this point.
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);
    
    // Compute output dimensions using the pooling formula
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    // Set block dimensions matching the defined tile sizes
    dim3 threads(TILE_W, TILE_H, TILE_D);
    
    // Grid dimensions: blockIdx.x for output width, blockIdx.y for output height, and blockIdx.z combines (batch, channel, depth tile)
    int grid_x = (out_w + TILE_W - 1) / TILE_W;
    int grid_y = (out_h + TILE_H - 1) / TILE_H;
    int d_tiles = (out_d + TILE_D - 1) / TILE_D;
    int grid_z = batch_size * channels * d_tiles;
    dim3 blocks(grid_x, grid_y, grid_z);

    // Compute the shared memory size per block
    int tile_w_eff = (TILE_W - 1) * stride + kernel_size;
    int tile_h_eff = (TILE_H - 1) * stride + kernel_size;
    int tile_d_eff = (TILE_D - 1) * stride + kernel_size;
    size_t shared_mem_size = tile_w_eff * tile_h_eff * tile_d_eff * sizeof(float);

    avg_pool3d_shared_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) - shared memory optimized");
}
