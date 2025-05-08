#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

// Tiling dimensions for shared memory kernel
#define TILE_D 4
#define TILE_H 8
#define TILE_W 8

// Kernel using shared memory to load input tiles for 3D max pooling
// Each block processes a tile from one (b, c) slice
// Grid dimensions: 
//   grid.x = number of tiles in output width,
//   grid.y = number of tiles in output height,
//   grid.z = (batch_size * channels * number of tiles in output depth)

template <typename scalar_t>
__global__ void max_pool3d_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    // Calculate number of depth tiles in the output
    int num_tiles_d = (output_d + TILE_D - 1) / TILE_D;

    // Decompose grid.z into (b, c) and tile index in depth:
    int tile_d_idx = blockIdx.z % num_tiles_d;  // which tile (in depth) within the (b,c) slice
    int bc_idx = blockIdx.z / num_tiles_d;         // index for combined (b, c)
    int b = bc_idx / channels;
    int c = bc_idx % channels;

    // Determine the starting output indices for this block tile
    int out_w_tile_start = blockIdx.x * TILE_W;
    int out_h_tile_start = blockIdx.y * TILE_H;
    int out_d_tile_start = tile_d_idx * TILE_D;

    // Corresponding starting global input coordinates for the tile
    int in_d0 = out_d_tile_start * stride - padding;
    int in_h0 = out_h_tile_start * stride - padding;
    int in_w0 = out_w_tile_start * stride - padding;

    // Dimensions of the input tile to load in shared memory
    int tile_in_d = TILE_D * stride + (kernel_size - 1) * dilation;
    int tile_in_h = TILE_H * stride + (kernel_size - 1) * dilation;
    int tile_in_w = TILE_W * stride + (kernel_size - 1) * dilation;
    int smem_size = tile_in_d * tile_in_h * tile_in_w;  // total number of elements in shared memory

    extern __shared__ char shared_data[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(shared_data);

    // Compute base offset in the input for this (b, c) slice
    int input_channel_offset = ((b * channels + c) * input_d * input_h * input_w);

    // Load the shared memory tile cooperatively
    int block_threads = blockDim.x * blockDim.y * blockDim.z;
    int thread_id = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
    
    for (int s = thread_id; s < smem_size; s += block_threads) {
        int d_local = s / (tile_in_h * tile_in_w);
        int rem = s % (tile_in_h * tile_in_w);
        int h_local = rem / tile_in_w;
        int w_local = rem % tile_in_w;
        
        int g_d = in_d0 + d_local;
        int g_h = in_h0 + h_local;
        int g_w = in_w0 + w_local;

        scalar_t value;
        if (g_d < 0 || g_d >= input_d || g_h < 0 || g_h >= input_h || g_w < 0 || g_w >= input_w) {
            value = -std::numeric_limits<scalar_t>::infinity();
        } else {
            int global_input_idx = input_channel_offset + (g_d * input_h * input_w + g_h * input_w + g_w);
            value = input[global_input_idx];
        }
        smem[s] = value;
    }

    __syncthreads();

    // Each thread now computes one output element in the tile.
    // The block is launched with block dimensions: (TILE_W, TILE_H, TILE_D)
    int local_w = threadIdx.x;  // [0, TILE_W)
    int local_h = threadIdx.y;  // [0, TILE_H)
    int local_d = threadIdx.z;  // [0, TILE_D)

    // Compute global output coordinates
    int out_w_idx = out_w_tile_start + local_w;
    int out_h_idx = out_h_tile_start + local_h;
    int out_d_idx = out_d_tile_start + local_d;

    if (out_w_idx >= output_w || out_h_idx >= output_h || out_d_idx >= output_d)
        return;

    // For each output, the pooling window in shared memory starts at:
    int start_w = local_w * stride;
    int start_h = local_h * stride;
    int start_d = local_d * stride;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int best_kd = 0, best_kh = 0, best_kw = 0;

    // Iterate over the pooling kernel window
    for (int kd = 0; kd < kernel_size; ++kd) {
        int d_index = start_d + kd * dilation;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_index = start_h + kh * dilation;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int w_index = start_w + kw * dilation;
                int smem_index = d_index * (tile_in_h * tile_in_w) + h_index * tile_in_w + w_index;
                scalar_t val = smem[smem_index];
                if (val > max_val) {
                    max_val = val;
                    best_kd = kd;
                    best_kh = kh;
                    best_kw = kw;
                }
            }
        }
    }

    // Compute output global index and write output
    int output_idx = (((b * channels + c) * output_d + out_d_idx) * output_h + out_h_idx) * output_w + out_w_idx;
    output[output_idx] = max_val;

    if (indices != nullptr) {
        // Compute global input coordinates corresponding to the max value
        int best_g_d = in_d0 + start_d + best_kd * dilation;
        int best_g_h = in_h0 + start_h + best_kh * dilation;
        int best_g_w = in_w0 + start_w + best_kw * dilation;
        int global_index = input_channel_offset + (best_g_d * input_h * input_w + best_g_h * input_w + best_g_w);
        indices[output_idx] = global_index;
    }
}


// Host function to launch the shared memory kernel

torch::Tensor max_pool3d_cuda_forward_shared(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode) {

    // Input dimensions: [B, C, D, H, W]
    auto input_sizes = input.sizes();
    const int batch_size = input_sizes[0];
    const int channels = input_sizes[1];
    const int input_d = input_sizes[2];
    const int input_h = input_sizes[3];
    const int input_w = input_sizes[4];

    // Compute output dimensions
    float d_out_float = ((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / (float)stride) + 1;
    float h_out_float = ((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / (float)stride) + 1;
    float w_out_float = ((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / (float)stride) + 1;
    
    int output_d = ceil_mode ? std::ceil(d_out_float) : std::floor(d_out_float);
    int output_h = ceil_mode ? std::ceil(h_out_float) : std::floor(h_out_float);
    int output_w = ceil_mode ? std::ceil(w_out_float) : std::floor(w_out_float);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ?
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();
    
    // Determine grid dimensions based on tiling for one (b, c) slice
    int num_tiles_w = (output_w + TILE_W - 1) / TILE_W;
    int num_tiles_h = (output_h + TILE_H - 1) / TILE_H;
    int num_tiles_d = (output_d + TILE_D - 1) / TILE_D;

    // Grid: x -> tiles in width, y -> tiles in height, z -> (batch*channels*num_tiles_d)
    dim3 grid(num_tiles_w, num_tiles_h, batch_size * channels * num_tiles_d);
    // Block dimensions: one thread per output element in the tile
    dim3 block(TILE_W, TILE_H, TILE_D);

    // Compute shared memory size
    int tile_in_d = TILE_D * stride + (kernel_size - 1) * dilation;
    int tile_in_h = TILE_H * stride + (kernel_size - 1) * dilation;
    int tile_in_w = TILE_W * stride + (kernel_size - 1) * dilation;
    size_t shared_mem_size = tile_in_d * tile_in_h * tile_in_w * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_forward_cuda_shared", ([&] {
        max_pool3d_shared_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size,
            channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size, stride, padding, dilation);
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool3d_cuda_forward_shared, "3D Max Pooling forward with shared memory (CUDA)");
}
