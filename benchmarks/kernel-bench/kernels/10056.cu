#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile sizes for output (each block computes an 8x8 tile of the output)
#define TILE_OUT_H 8
#define TILE_OUT_W 8

// Shared memory tiled kernel for depthwise convolution
__global__ void depthwise_conv2d_shared(
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
    // Determine the output tile start indices (in global output space)
    int tile_out_row = blockIdx.y * TILE_OUT_H;
    int tile_out_col = blockIdx.x * TILE_OUT_W;

    // blockIdx.z indexes a specific combination of batch and output channel
    int index = blockIdx.z;
    int c_out = index % out_channels;
    int b = index / out_channels;

    // Compute group and filter index within the group
    int g = c_out / channels_per_group;
    int m = c_out % channels_per_group;  // used for weight indexing

    // Compute the dimensions of the required input tile for this output tile
    // Formula: tile_in_dim = (TILE_OUT - 1) * stride + (kernel - 1) * dilation + 1
    int tile_in_h = (TILE_OUT_H - 1) * stride_h + (kernel_h - 1) * dilation_h + 1;
    int tile_in_w = (TILE_OUT_W - 1) * stride_w + (kernel_w - 1) * dilation_w + 1;

    // The top-left global input coordinate for the tile
    int in_tile_row = tile_out_row * stride_h - padding_h;
    int in_tile_col = tile_out_col * stride_w - padding_w;

    // Allocate dynamic shared memory for the input tile
    extern __shared__ float shmem[];   // size = tile_in_h * tile_in_w
    int total_shm = tile_in_h * tile_in_w;

    // Each thread in the block loads some elements from global memory
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    for (int i = thread_id; i < total_shm; i += num_threads) {
        int sh_r = i / tile_in_w;
        int sh_c = i % tile_in_w;
        int global_r = in_tile_row + sh_r;
        int global_c = in_tile_col + sh_c;
        float val = 0.0f;
        if (global_r >= 0 && global_r < in_h && global_c >= 0 && global_c < in_w) {
            // Input channel for this block is g.
            int input_idx = ((b * in_channels + g) * in_h + global_r) * in_w + global_c;
            val = input[input_idx];
        }
        shmem[i] = val;
    }
    __syncthreads();  // synchronize once after loading the shared tile

    // Now, each thread computes one output element in the tile
    int local_out_row = threadIdx.y;
    int local_out_col = threadIdx.x;
    if (local_out_row < TILE_OUT_H && local_out_col < TILE_OUT_W) {
        int out_r = tile_out_row + local_out_row;
        int out_c = tile_out_col + local_out_col;
        if (out_r < out_h && out_c < out_w) {
            float sum = 0.0f;
            // The starting point in shared memory for this output's receptive field
            int shm_start_r = local_out_row * stride_h;
            int shm_start_c = local_out_col * stride_w;
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int sh_r = shm_start_r + kh * dilation_h;
                    int sh_c = shm_start_c + kw * dilation_w;
                    if (sh_r < tile_in_h && sh_c < tile_in_w) {
                        float in_val = shmem[sh_r * tile_in_w + sh_c];
                        int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                        float w_val = weight[weight_idx];
                        sum += in_val * w_val;
                    }
                }
            }
            if (bias != nullptr) {
                sum += bias[c_out];
            }
            int out_idx = ((b * out_channels + c_out) * out_h + out_r) * out_w + out_c;
            output[out_idx] = sum;
        }
    }
}

// Host function to launch the shared memory tiled depthwise convolution kernel
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
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
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

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    // Set up grid dimensions. Each block covers a TILE_OUT_H x TILE_OUT_W output tile for one (batch, channel)
    int grid_x = (out_w + TILE_OUT_W - 1) / TILE_OUT_W;
    int grid_y = (out_h + TILE_OUT_H - 1) / TILE_OUT_H;
    int grid_z = batch_size * out_channels;
    
    dim3 threads(TILE_OUT_W, TILE_OUT_H);  // one thread computes one output element
    dim3 blocks(grid_x, grid_y, grid_z);

    // Compute shared memory size needed per block
    int tile_in_h = (TILE_OUT_H - 1) * stride_h + (kernel_h - 1) * dilation_h + 1;
    int tile_in_w = (TILE_OUT_W - 1) * stride_w + (kernel_w - 1) * dilation_w + 1;
    size_t shared_mem_size = tile_in_h * tile_in_w * sizeof(float);

    depthwise_conv2d_shared<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "Shared Memory Tiled Depthwise Conv2D forward (CUDA)");
}
