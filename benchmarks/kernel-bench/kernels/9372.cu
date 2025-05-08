#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constants for kernel configuration
#define CHANNELS_PER_BLOCK 4
#define TILE_IC 8  // Number of input channels per tile

// The kernel overlaps computation with memory transfers by asynchronously prefetching weight tiles
// into shared memory using the cp.async instruction with a double buffering scheme.

// Note: This kernel relies on CUDA 12.2 and NVIDIA H100 architecture supporting cp.async.bulk.shared.global.

__global__ void conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int height_out,
    int width_out,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {

    // Compute thread and block indices for output pixel
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int blockDimX = blockDim.x; // e.g., 16
    int blockDimY = blockDim.y; // e.g., 16
    int h_out = by * blockDimY + ty;
    int w_out = bx * blockDimX + tx;

    // Decode batch and output channel group from blockIdx.z
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int b = bz / groups_per_batch;
    int g = bz % groups_per_batch;
    int oc_start = g * CHANNELS_PER_BLOCK;

    if (h_out >= height_out || w_out >= width_out || b >= batch_size)
        return;

    // Initialize accumulation registers with bias
    float sums[CHANNELS_PER_BLOCK];
#pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        int global_oc = oc_start + i;
        sums[i] = (global_oc < out_channels && bias != nullptr) ? bias[global_oc] : 0.0f;
    }

    int kernel_area = kernel_h * kernel_w;
    // Calculate shared memory buffer size per tile
    // Each tile: CHANNELS_PER_BLOCK * TILE_IC * kernel_area floats
    const int tile_buffer_size = CHANNELS_PER_BLOCK * TILE_IC * kernel_area;
    // We use double buffering, so total shared memory allocated is 2 * tile_buffer_size floats
    extern __shared__ float shared_mem[]; // size: 2 * tile_buffer_size * sizeof(float)

    // Determine number of tiles to cover all input channels
    int num_tiles = (in_channels + TILE_IC - 1) / TILE_IC;
    int cur_buf = 0; // which buffer (0 or 1) holds the current weight tile

    // Preload first tile (tile index 0) into buffer cur_buf using asynchronous copy
    {
        int tile_ic_start = 0;
        int current_tile = (in_channels - tile_ic_start) < TILE_IC ? (in_channels - tile_ic_start) : TILE_IC;
        int tile_elems = CHANNELS_PER_BLOCK * current_tile * kernel_area;
        int lane = ty * blockDimX + tx;
        int blockSize = blockDimX * blockDimY;
        for (int idx = lane; idx < tile_elems; idx += blockSize) {
            // Compute indices within the tile
            int local_oc = idx / (current_tile * kernel_area);          // Which output channel in the group
            int rem = idx % (current_tile * kernel_area);
            int local_ic = rem / kernel_area;                              // Which input channel within the tile
            int k = rem % kernel_area;                                     // Kernel element index

            int global_oc = oc_start + local_oc;
            // Compute global weight offset. Weight layout is [out_channels, in_channels, kernel_h, kernel_w]
            int global_idx = global_oc * (in_channels * kernel_area) + (tile_ic_start + local_ic) * kernel_area + k;
            // Asynchronously copy one float (4 bytes) from global memory to shared memory
            asm volatile("cp.async.bulk.shared.global [%0], [%1], %2;\n"
                          :
                          : "r"(shared_mem + cur_buf * tile_buffer_size + idx),
                            "l"(weight + global_idx),
                            "n"(4));
        }
        // Wait until the cp.async operations for the first tile have completed
        __syncthreads();
    }

    // Loop over all input channel tiles with double buffering
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_ic_start = tile * TILE_IC;
        int current_tile = (in_channels - tile_ic_start) < TILE_IC ? (in_channels - tile_ic_start) : TILE_IC;
        int tile_elems = CHANNELS_PER_BLOCK * current_tile * kernel_area;

        // If there is a next tile, start asynchronous copy into the alternate buffer
        if (tile < num_tiles - 1) {
            int next_tile_ic_start = (tile + 1) * TILE_IC;
            int next_tile = (in_channels - next_tile_ic_start) < TILE_IC ? (in_channels - next_tile_ic_start) : TILE_IC;
            int next_tile_elems = CHANNELS_PER_BLOCK * next_tile * kernel_area;
            int lane = ty * blockDimX + tx;
            int blockSize = blockDimX * blockDimY;
            int next_buf = 1 - cur_buf;
            for (int idx = lane; idx < next_tile_elems; idx += blockSize) {
                int local_oc = idx / (next_tile * kernel_area);
                int rem = idx % (next_tile * kernel_area);
                int local_ic = rem / kernel_area;
                int k = rem % kernel_area;
                int global_oc = oc_start + local_oc;
                int global_idx = global_oc * (in_channels * kernel_area) + (next_tile_ic_start + local_ic) * kernel_area + k;
                asm volatile("cp.async.bulk.shared.global [%0], [%1], %2;\n"
                              :
                              : "r"(shared_mem + next_buf * tile_buffer_size + idx),
                                "l"(weight + global_idx),
                                "n"(4));
            }
        }

        // Wait until the asynchronous copies for the current tile are visible
        __syncthreads();

        // For the current tile, loop over the input channels in this tile
        for (int local_ic = 0; local_ic < current_tile; local_ic++) {
            int ic = tile_ic_start + local_ic;
            for (int kh = 0; kh < kernel_h; kh++) {
                int h_in = h_out * stride + kh * dilation_h - pad_h;
                if (h_in < 0 || h_in >= input_height) continue;
                for (int kw = 0; kw < kernel_w; kw++) {
                    int w_in = w_out * stride + kw * dilation_w - pad_w;
                    if (w_in < 0 || w_in >= input_width) continue;
                    float x_val = __ldg(&x[b * in_channels * input_height * input_width +
                                           ic * (input_height * input_width) +
                                           h_in * input_width + w_in]);
                    // Use the weights from the current shared memory buffer
                    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
                        int weight_offset = i * (TILE_IC * kernel_area) + local_ic * kernel_area + (kh * kernel_w + kw);
                        float w_val = shared_mem[cur_buf * tile_buffer_size + weight_offset];
                        sums[i] += x_val * w_val;
                    }
                }
            }
        }

        // Swap buffers if there is a next tile
        if (tile < num_tiles - 1) {
            cur_buf = 1 - cur_buf;
        }
    }

    // Write results to global memory output tensor
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        int global_oc = oc_start + i;
        if (global_oc < out_channels) {
            int out_idx = b * (out_channels * height_out * width_out) +
                          global_oc * (height_out * width_out) +
                          h_out * width_out + w_out;
            output[out_idx] = sums[i];
        }
    }
}

// The forward function wraps the kernel launch and is exposed to PyTorch via pybind11

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,  // bias is optional
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        bias_ptr = bias->data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);

    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    // Define block and grid dimensions
    const int BLOCK_SIZE_X = 16;
    const int BLOCK_SIZE_Y = 16;
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    dim3 blocks((width_out + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (height_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
                batch_size * groups_per_batch);

    // Shared memory size: 2 * (CHANNELS_PER_BLOCK * TILE_IC * kernel_h * kernel_w) floats
    size_t shared_mem_size = 2 * CHANNELS_PER_BLOCK * TILE_IC * kernel_h * kernel_w * sizeof(float);

    conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        height_out,
        width_out,
        stride,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv2D forward (CUDA) with overlapped async weight prefetching");
}
