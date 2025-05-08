#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions for the output tile
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// This kernel maps the 4D output tensor (batch, channel, out_y, out_x) onto a 2D grid
// by combining the batch and channel dimensions into the grid.y dimension. This avoids
// potential underutilization issues related to a 3D grid and provides a more balanced
// mapping of threads to the spatial domain. Shared memory is used to load the input patch
// and the small filter (kernel) for each (batch, channel) pair.
__global__ void depthwise_conv2d_index2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    // Calculate number of tiles in the vertical output dimension
    int numTilesY = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;

    // Determine the starting x-coordinate of this output tile
    int tile_out_x = blockIdx.x * TILE_WIDTH;
    // For the y-dimension, the blockIdx.y is remapped as follows:
    // blockIdx.y = (tile index in y) + (combined index of batch and channel)*numTilesY
    int tile_out_y = (blockIdx.y % numTilesY) * TILE_HEIGHT;
    int bc_index = blockIdx.y / numTilesY;  // This indexes the combined (batch, out_channel) pair
    int b = bc_index / out_channels;
    int oc = bc_index % out_channels;

    // Map the output channel to corresponding input channel and weight group
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Each thread in the block computes one output pixel in the tile
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int out_x = tile_out_x + thread_x;
    int out_y = tile_out_y + thread_y;

    // Compute the starting coordinate in the input that corresponds to the top-left
    // of the required patch for the current output tile
    int in_start_x = tile_out_x * stride - padding;
    int in_start_y = tile_out_y * stride - padding;

    // Dimensions of the input patch that must be loaded into shared memory
    int smem_cols = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_rows = (TILE_HEIGHT - 1) * stride + kernel_size;

    extern __shared__ float shared_mem[];
    // First part of shared memory holds the input patch
    float* s_input = shared_mem;
    // Second part holds the kernel filter
    float* s_weight = shared_mem + smem_rows * smem_cols;

    // Use a linear index for cooperative loading by the block
    int linear_thread = thread_y * blockDim.x + thread_x;
    int block_threads = blockDim.x * blockDim.y;

    // Load the kernel weights for this (in_ch, weight_ch) into shared memory
    int num_weight = kernel_size * kernel_size;
    for (int i = linear_thread; i < num_weight; i += block_threads) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) + i
        ];
    }

    // Load the input patch into shared memory
    int total_input = smem_rows * smem_cols;
    for (int i = linear_thread; i < total_input; i += block_threads) {
        int r = i / smem_cols;
        int c = i % smem_cols;
        int global_y = in_start_y + r;
        int global_x = in_start_x + c;
        float val = 0.0f;
        if (global_y >= 0 && global_y < input_h && global_x >= 0 && global_x < input_w) {
            int input_idx = b * (in_channels * input_h * input_w) +
                            in_ch * (input_h * input_w) +
                            global_y * input_w + global_x;
            val = input[input_idx];
        }
        s_input[i] = val;
    }

    __syncthreads();

    // Compute the convolution for the output pixel if it falls within bounds
    if (out_x < output_w && out_y < output_h) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int s_y = thread_y * stride + ky;
                int s_x = thread_x * stride + kx;
                float in_val = s_input[s_y * smem_cols + s_x];
                float wt = s_weight[ky * kernel_size + kx];
                sum += in_val * wt;
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int out_idx = b * (out_channels * output_h * output_w) +
                      oc * (output_h * output_w) +
                      out_y * output_w + out_x;
        output[out_idx] = sum;
    }
}

// The forward function exposed to Python via pybind11.
// It computes the output dimensions, sets up the grid and block dimensions using a 2D grid (folding batch and channel into grid.y),
// and launches the kernel with dynamic shared memory that accommodates both the input patch and filter weights.

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Input and weight must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Grid dimensions:
    // grid.x covers the horizontal output tiles
    int numTilesX = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
    // grid.y combines vertical tiles and the (batch, channel) dimensions
    int numTilesY = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_y = numTilesY * (batch_size * out_channels);
    dim3 grid(numTilesX, grid_y);
    dim3 block(TILE_WIDTH, TILE_HEIGHT);

    // Calculate the required shared memory size per block
    int smem_cols = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_rows = (TILE_HEIGHT - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (smem_rows * smem_cols + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_index2d_kernel<<<grid, block, shared_mem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        output_h,
        output_w,
        kernel_size,
        stride,
        padding,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution with 2D Grid Indexing (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
