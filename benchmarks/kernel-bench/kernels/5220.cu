#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a tiling strategy with shared memory. Each block handles a tile of consecutive output indices
// for a specific (batch, channel) pair. The necessary contiguous input segment for the tile is loaded into
// shared memory to ensure coalesced accesses and evenly distribute work across threads and blocks.

__global__ void max_pool1d_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length,
    const bool return_indices) {

    // Each block in the x-dimension handles one (batch, channel) pair
    int bc = blockIdx.x;  // bc in [0, batch_size*num_channels)
    int b = bc / num_channels;
    int c = bc % num_channels;

    // Each block in the y-dimension processes a tile of consecutive output indices
    int tile_start = blockIdx.y * blockDim.x;  // starting output index for this tile
    int out_idx = tile_start + threadIdx.x;      // output index computed by this thread

    // Compute the input range needed for this tile
    // For an output index o, the pooling window covers [o*stride - padding, o*stride - padding + (kernel_size-1)*dilation]
    // For the tile, the range is from:
    // tile_input_start = tile_start*stride - padding
    // to
    // tile_input_end = (tile_start + blockDim.x - 1)*stride - padding + (kernel_size - 1)*dilation
    int tile_input_start = tile_start * stride - padding;
    int tile_input_end = (tile_start + blockDim.x - 1) * stride - padding + (kernel_size - 1) * dilation;
    int tile_width = tile_input_end - tile_input_start + 1;

    extern __shared__ float sdata[];

    // Load the necessary input tile into shared memory
    // Each thread loads multiple elements if needed
    for (int i = threadIdx.x; i < tile_width; i += blockDim.x) {
        int input_idx = tile_input_start + i;
        float val;
        if (input_idx < 0 || input_idx >= input_length) {
            // Out-of-bound positions are set to -INFINITY
            val = -INFINITY;
        } else {
            int global_input_idx = b * num_channels * input_length + c * input_length + input_idx;
            val = input[global_input_idx];
        }
        sdata[i] = val;
    }
    __syncthreads();

    if (out_idx < output_length) {
        // For this output index, compute the corresponding start in global input
        int out_start = out_idx * stride - padding;
        // Compute relative offset in shared memory
        int shared_offset = out_start - tile_input_start;
        float max_val = -INFINITY;
        int max_index = -1;

        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int offset = shared_offset + k * dilation;
            if (offset >= 0 && offset < tile_width) {
                float val = sdata[offset];
                if (val > max_val) {
                    max_val = val;
                    max_index = out_start + k * dilation;
                }
            }
        }

        int global_out_idx = b * num_channels * output_length + c * output_length + out_idx;
        output[global_out_idx] = max_val;
        if (return_indices) {
            indices[global_out_idx] = max_index;
        }
    }
}

// Host forward function
torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {

    TORCH_CHECK(x.dim() == 3, "Input must be 3D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int batch_size = x.size(0);
    const int num_channels = x.size(1);
    const int input_length = x.size(2);

    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    TORCH_CHECK(output_length > 0, "Output length must be positive");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, num_channels, output_length}, options);
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
    }

    // Set tile size: number of output elements computed per block (per (batch, channel) pair)
    int tile_size = 128;
    dim3 threads(tile_size);
    int tiles = (output_length + tile_size - 1) / tile_size;
    // Grid: one block for each (batch, channel) pair per tile
    dim3 blocks(batch_size * num_channels, tiles);

    // Compute shared memory size per block:
    // tile_width = (tile_size - 1) * stride + (kernel_size - 1) * dilation + 1
    int tile_width = (tile_size - 1) * stride + (kernel_size - 1) * dilation + 1;
    size_t shared_mem_size = tile_width * sizeof(float);

    max_pool1d_shared_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<int64_t>() : nullptr,
        batch_size,
        num_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        output_length,
        return_indices
    );

    return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaxPool1D forward with even workload distribution using shared memory (CUDA)");
}
