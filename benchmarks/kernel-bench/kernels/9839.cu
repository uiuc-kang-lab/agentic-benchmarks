#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions for the output tile processed by each block
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// This kernel computes a depthwise convolution by partitioning the output into tiles.  
// Within each tile, the convolution reduction (over the kernel window) is parallelized across threads
// using atomicAdd in shared memory to safely accumulate partial products.  
// By confining atomic operations to fast shared memory and having each block write to a unique portion
// of the output tensor, we minimize global memory atomics and reduce contention.
__global__ void depthwise_conv2d_atomic_tile_kernel(
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
    // Determine the batch and output channel that this block is processing
    int block_id = blockIdx.z;
    int b = block_id / out_channels;
    int oc = block_id % out_channels;
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Determine the top-left corner (in output space) of the tile for this block
    int tile_out_x = blockIdx.x * TILE_WIDTH;
    int tile_out_y = blockIdx.y * TILE_HEIGHT;

    // Allocate shared memory: first for accumulating the tile outputs, then for the weight tile
    extern __shared__ float shared_mem[];
    // tile_output holds the partial sums for the output tile (size: TILE_WIDTH * TILE_HEIGHT)
    float* tile_output = shared_mem;
    // s_weight holds the filter kernel in shared memory (size: kernel_size * kernel_size)
    float* s_weight = tile_output + TILE_WIDTH * TILE_HEIGHT;

    // Initialize the tile output shared memory to 0
    int tile_size = TILE_WIDTH * TILE_HEIGHT;
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        tile_output[i] = 0.0f;
    }

    // Load the weight tile into shared memory (each block uses its own filter for the given channel)
    int weight_size = kernel_size * kernel_size;
    if (threadIdx.x < weight_size) {
        int weight_index = in_ch * (channels_per_group * weight_size) +
                           weight_ch * weight_size + threadIdx.x;
        s_weight[threadIdx.x] = weight[weight_index];
    }
    __syncthreads();

    // Total work: each output pixel in the tile must accumulate contributions from all kernel elements
    int reduce_size = weight_size; // kernel_size * kernel_size
    int total_work = tile_size * reduce_size;

    // Distribute the reduction work among threads in the block
    for (int idx = threadIdx.x; idx < total_work; idx += blockDim.x) {
        int tile_pixel = idx / reduce_size;  // which output pixel in the tile
        int red_idx = idx % reduce_size;       // which kernel element (row-major order)

        // Compute the tile-local coordinates for this output pixel
        int out_local_y = tile_pixel / TILE_WIDTH;
        int out_local_x = tile_pixel % TILE_WIDTH;
        // Compute the corresponding global output coordinates
        int global_out_y = tile_out_y + out_local_y;
        int global_out_x = tile_out_x + out_local_x;

        // Only process if within output bounds
        if (global_out_y < output_h && global_out_x < output_w) {
            int k_y = red_idx / kernel_size;
            int k_x = red_idx % kernel_size;
            // Determine the corresponding input coordinate
            int in_y = global_out_y * stride + k_y - padding;
            int in_x = global_out_x * stride + k_x - padding;
            float partial = 0.0f;
            if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
                int input_index = b * (in_channels * input_h * input_w) + 
                                  in_ch * (input_h * input_w) + 
                                  in_y * input_w + in_x;
                partial = input[input_index] * s_weight[red_idx];
            }
            // Atomically accumulate the partial result into the tile output in shared memory
            atomicAdd(&tile_output[tile_pixel], partial);
        }
    }
    __syncthreads();

    // Write the computed tile back to global memory
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        int out_local_y = i / TILE_WIDTH;
        int out_local_x = i % TILE_WIDTH;
        int global_out_y = tile_out_y + out_local_y;
        int global_out_x = tile_out_x + out_local_x;
        if (global_out_y < output_h && global_out_x < output_w) {
            float out_val = tile_output[i];
            if (bias != nullptr) {
                out_val += bias[oc];
            }
            int output_index = b * (out_channels * output_h * output_w) +
                               oc * (output_h * output_w) +
                               global_out_y * output_w + global_out_x;
            output[output_index] = out_val;
        }
    }
}

// Forward function callable from Python
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
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

    // Each block processes one output tile for a given (batch, channel) pair
    dim3 grid((output_w + TILE_WIDTH - 1) / TILE_WIDTH,
              (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
              batch_size * out_channels);
    int block_size = 256;  // 1D block

    // Shared memory: space for tile_output and the weight tile
    size_t shared_mem_bytes = (TILE_WIDTH * TILE_HEIGHT + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    depthwise_conv2d_atomic_tile_kernel<<<grid, block_size, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Atomic Tile Reduction (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
