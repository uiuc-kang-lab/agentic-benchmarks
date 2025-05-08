#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constants for warp and tiling configuration
#define WARP_SIZE 32
// Define the tiling dimensions in terms of warps per block
#define TILE_COLS 2
#define TILE_ROWS 4
#define WARPS_PER_BLOCK (TILE_COLS * TILE_ROWS)  // 8 warps per block

// Block dimensions: one warp per output pixel computed cooperatively
#define BLOCK_DIM_X WARP_SIZE         // 32 threads (one warp) per output pixel
#define BLOCK_DIM_Y (WARPS_PER_BLOCK)   // Number of warps (i.e., output pixels computed per block)

// New Kernel: Each warp computes one output pixel using warp-level reduction.
// The grid is organized into a 3D grid: grid.z maps to (batch, channel) pairs, and grid.x and grid.y tile the output spatial dimensions.

__global__ void tiling_warp_depthwise_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    // Decode batch and output channel from grid.z
    int bc = blockIdx.z;  // bc ranges from 0 to batch_size * out_channels - 1
    int b = bc / out_channels;
    int c_out = bc % out_channels;

    // Each block tiles the output spatial (width, height) dimensions.
    // We assign each warp (i.e., each row in our block) to one output pixel within the tile.
    // Compute warp id within the block using threadIdx.y (each warp is one "row" of the block).
    int warp_id = threadIdx.y;  // ranges from 0 to (WARPS_PER_BLOCK - 1)
    
    // Decode the 2D tile position of this warp.
    int tile_col = warp_id % TILE_COLS;  // horizontal tile offset within the block
    int tile_row = warp_id / TILE_COLS;  // vertical tile offset within the block

    // Compute the output pixel coordinates for this warp
    int out_x = blockIdx.x * TILE_COLS + tile_col;
    int out_y = blockIdx.y * TILE_ROWS + tile_row;

    // Check for boundary conditions
    if (out_x >= out_w || out_y >= out_h) return;

    // Within each warp, threadIdx.x defines the lane (0 ... 31)
    int lane = threadIdx.x; 

    int kernel_size = kernel_h * kernel_w;
    float sum = 0.0f;

    // Distribute kernel element computations among threads in the warp
    for (int k = lane; k < kernel_size; k += WARP_SIZE) {
        int kh = k / kernel_w;
        int kw = k % kernel_w;
        int h_in = out_y * stride_h - padding_h + kh * dilation_h;
        int w_in = out_x * stride_w - padding_w + kw * dilation_w;
        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
            int g = c_out / channels_per_group;  // group index
            // Compute the input and weight indices as in the original kernels
            int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
            int weight_idx = ((g * channels_per_group + (c_out % channels_per_group)) * kernel_h + kh) * kernel_w + kw;
            sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Warp-level reduction using shuffle intrinsics
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // First lane writes the result
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        int out_index = ((b * out_channels + c_out) * out_h + out_y) * out_w + out_x;
        output[out_index] = sum;
    }
}


// Host function that sets up the grid and block dimensions and launches the kernel

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias tensor must be a CUDA tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());
    
    const float* bias_ptr = (bias.has_value()) ? bias->data_ptr<float>() : nullptr;

    // Set up grid dimensions
    // Grid.x and Grid.y tile the spatial output dimensions according to our TILE_COLS and TILE_ROWS
    dim3 grid((out_w + TILE_COLS - 1) / TILE_COLS,
              (out_h + TILE_ROWS - 1) / TILE_ROWS,
              batch_size * out_channels);

    // Block dimensions: BLOCK_DIM_X (warp size) and BLOCK_DIM_Y (number of warps per block)
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    tiling_warp_depthwise_conv_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group
    );

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled Depthwise Conv2D with warp-level reduction (CUDA)");
}
