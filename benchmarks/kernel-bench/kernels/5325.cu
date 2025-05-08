#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Define tile dimensions for output
#define TILE_W 16
#define TILE_H 16

// Kernel using shared memory to load the input tile that is reused by threads
// Each block processes a tile of output for a single (b, c) pair.

template <typename scalar_t>
__global__ void max_pool2d_kernel_shared(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int channels  // number of channels
) {
    // Decode batch and channel indices from blockIdx.z
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;

    // Compute per-channel offsets
    int input_channel_size = input_height * input_width;
    int output_channel_size = output_height * output_width;
    const int input_offset = b * channels * input_channel_size + c * input_channel_size;
    const int output_offset = b * channels * output_channel_size + c * output_channel_size;

    // Determine the starting output coordinates for this block
    int tile_out_row = blockIdx.y * TILE_H;
    int tile_out_col = blockIdx.x * TILE_W;

    // Corresponding starting input coordinate for the tile
    int input_tile_start_row = tile_out_row * stride - padding;
    int input_tile_start_col = tile_out_col * stride - padding;

    // Determine dimensions for the shared memory tile
    // Shared memory tile must cover all input pixels needed for the output tile
    int SM_H = TILE_H * stride + (kernel_size - 1) * dilation;
    int SM_W = TILE_W * stride + (kernel_size - 1) * dilation;

    extern __shared__ char smem[];
    scalar_t* s_data = reinterpret_cast<scalar_t*>(smem);

    int num_smem_elements = SM_H * SM_W;
    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    // Load the shared memory tile from global memory
    // Each thread cooperatively loads elements
    for (int i = t_idx; i < num_smem_elements; i += total_threads) {
        int sm_r = i / SM_W;
        int sm_c = i % SM_W;
        int in_r = input_tile_start_row + sm_r;
        int in_c = input_tile_start_col + sm_c;

        if (in_r >= 0 && in_r < input_height && in_c >= 0 && in_c < input_width) {
            int in_index = input_offset + in_r * input_width + in_c;
            s_data[i] = input[in_index];
        } else {
            s_data[i] = -std::numeric_limits<scalar_t>::infinity();
        }
    }
    __syncthreads();

    // Determine the output coordinate for this thread within the block tile
    int out_r = tile_out_row + threadIdx.y;
    int out_c = tile_out_col + threadIdx.x;

    // Only compute if within output bounds
    if (out_r < output_height && out_c < output_width) {
        // Calculate local offsets within the shared memory tile
        // For the output element at (out_r, out_c), the corresponding top-left corner in shared memory is:
        // local_y = (threadIdx.y * stride) and local_x = (threadIdx.x * stride)
        int local_y = threadIdx.y * stride;
        int local_x = threadIdx.x * stride;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        // Iterate over the pooling window
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int s_r = local_y + kh * dilation;
                int s_c = local_x + kw * dilation;
                int s_index = s_r * SM_W + s_c;
                max_val = max(max_val, s_data[s_index]);
            }
        }
        int out_index = output_offset + out_r * output_width + out_c;
        output[out_index] = max_val;
    }
}


// Host function that prepares kernel launch parameters and invokes the kernel

torch::Tensor max_pool2d_cuda_forward_shared(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Input shape: [batch, channels, input_height, input_width]
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    // Compute output dimensions
    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Grid dims: each block processes a TILE_H x TILE_W region of output for a single (b, c) pair
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim(
        (output_width + TILE_W - 1) / TILE_W,
        (output_height + TILE_H - 1) / TILE_H,
        batch_size * channels
    );

    // Determine shared memory size needed per block
    int SM_H = TILE_H * stride + (kernel_size - 1) * dilation;
    int SM_W = TILE_W * stride + (kernel_size - 1) * dilation;
    size_t shared_mem_size = SM_H * SM_W * sizeof(at::Half); // dummy type

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward_shared", ([&] {
        shared_mem_size = SM_H * SM_W * sizeof(scalar_t);
        max_pool2d_kernel_shared<scalar_t><<<gridDim, blockDim, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation,
            channels
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_shared, "Max Pool 2D forward with shared memory (CUDA)");
}
