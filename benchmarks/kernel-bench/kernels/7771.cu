#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE 16
#define TC 8

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses shared memory tiling to load the input tile once and reuse it for multiple output computations.
// It loops over input channels in tiles of size TC. __syncthreads() is called only after loading each tile
// and after computation to ensure shared memory consistency when reusing the buffer.

__global__ void tile_shared_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, // can be nullptr
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_height,
    int kernel_width,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Declare external shared memory
    extern __shared__ float shared[];

    // Determine the top-left coordinates of the output tile
    int tile_out_x = blockIdx.x * TILE;
    int tile_out_y = blockIdx.y * TILE;

    // Thread coordinates within the tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global output coordinates
    int out_x = tile_out_x + tx;
    int out_y = tile_out_y + ty;

    // Determine batch and output channel from blockIdx.z
    int linear_idx = blockIdx.z;
    int b = linear_idx / out_channels;
    int oc = linear_idx % out_channels;

    // Determine grouping
    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = oc / out_channels_per_group;

    float sum = 0.0f;

    // Compute required input tile dimensions
    int tile_in_h = TILE * stride + (kernel_height - 1) * dilation;
    int tile_in_w = TILE * stride + (kernel_width - 1) * dilation;

    // Top-left coordinate in the input corresponding to this output tile
    int in_tile_x = tile_out_x * stride - padding;
    int in_tile_y = tile_out_y * stride - padding;

    // Loop over input channels (for this group) in tiles of TC channels
    for (int c_tile = 0; c_tile < in_channels_per_group; c_tile += TC) {
        int current_tile_channels = (c_tile + TC <= in_channels_per_group) ? TC : (in_channels_per_group - c_tile);
        int num_shm = current_tile_channels * tile_in_h * tile_in_w;

        // Each thread loads multiple elements into shared memory
        int tid = ty * TILE + tx; 
        for (int i = tid; i < num_shm; i += TILE * TILE) {
            int local_ch = i / (tile_in_h * tile_in_w);
            int rem = i % (tile_in_h * tile_in_w);
            int i_row = rem / tile_in_w;
            int i_col = rem % tile_in_w;
            int global_y = in_tile_y + i_row;
            int global_x = in_tile_x + i_col;
            float val = 0.0f;
            int c = group * in_channels_per_group + c_tile + local_ch;
            if (global_y >= 0 && global_y < in_height && global_x >= 0 && global_x < in_width) {
                int input_idx = ((b * in_channels + c) * in_height + global_y) * in_width + global_x;
                val = input[input_idx];
            }
            shared[i] = val;
        }
        __syncthreads();  // Synchronize after shared memory load

        // Compute convolution using the loaded input tile for this channel block
        for (int local_ch = 0; local_ch < current_tile_channels; local_ch++) {
            // Global channel index
            int c = group * in_channels_per_group + c_tile + local_ch;
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int shm_y = ty * stride + kh * dilation;
                    int shm_x = tx * stride + kw * dilation;
                    if (shm_y < tile_in_h && shm_x < tile_in_w) {
                        int shm_index = local_ch * (tile_in_h * tile_in_w) + shm_y * tile_in_w + shm_x;
                        float in_val = shared[shm_index];
                        int weight_idx = ((oc * in_channels_per_group + (c_tile + local_ch)) * kernel_height + kh) * kernel_width + kw;
                        float w_val = weight[weight_idx];
                        sum += in_val * w_val;
                    }
                }
            }
        }
        __syncthreads(); // Synchronize before loading the next channel tile
    }

    // Write output if within valid boundaries
    if (out_x < out_width && out_y < out_height) {
        int output_idx = ((b * out_channels + oc) * out_height + out_y) * out_width + out_x;
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[output_idx] = sum;
    }
}


// The forward function sets up the kernel launch configuration and computes the output dimensions.
// It calculates the shared memory size based on the tile and channel tile sizes, and launches the kernel with minimal synchronizations.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    // Compute output spatial dimensions
    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    // Define grid dimensions: grid.x covers output width tiles, grid.y covers output height tiles, grid.z covers batch * out_channels
    int grid_x = (out_width + TILE - 1) / TILE;
    int grid_y = (out_height + TILE - 1) / TILE;
    int grid_z = batch_size * out_channels;
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(TILE, TILE);

    // Compute shared memory size: each channel tile requires TC * tile_in_h * tile_in_w floats
    int tile_in_h = TILE * stride + (kernel_height - 1) * dilation;
    int tile_in_w = TILE * stride + (kernel_width - 1) * dilation;
    int shared_mem_size = TC * tile_in_h * tile_in_w * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "tile_shared_conv2d_forward_cuda", ([&] {
        tile_shared_conv2d_kernel<<<grid, block, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_height,
            kernel_width,
            out_height,
            out_width,
            stride,
            padding,
            dilation,
            groups);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tile Shared Memory CUDA 2D Convolution");
}
