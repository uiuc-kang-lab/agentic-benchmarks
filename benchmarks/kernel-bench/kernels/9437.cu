#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions and number of output channels processed per block in z-dimension
#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define NUM_OC_PER_BLOCK 4

// CUDA kernel using shared memory with minimal synchronizations for convolution
__global__ void conv2d_kernel_shared(
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

    // Determine which batch and output channel this block corresponds to
    int num_oc_tiles = (out_channels + NUM_OC_PER_BLOCK - 1) / NUM_OC_PER_BLOCK;
    int b = blockIdx.z / num_oc_tiles;
    int oc_base = (blockIdx.z % num_oc_tiles) * NUM_OC_PER_BLOCK;
    int oc = oc_base + threadIdx.z;

    // Determine the block's top-left output coordinate
    int block_out_y = blockIdx.y * TILE_HEIGHT;
    int block_out_x = blockIdx.x * TILE_WIDTH;
    int out_y = block_out_y + threadIdx.y;
    int out_x = block_out_x + threadIdx.x;

    if (b >= batch_size || out_y >= height_out || out_x >= width_out || oc >= out_channels)
        return;

    // Initialize accumulator with bias if provided
    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Compute shared memory tile dimensions
    int sh_width = TILE_WIDTH * stride + (kernel_w - 1) * dilation_w;
    int sh_height = TILE_HEIGHT * stride + (kernel_h - 1) * dilation_h;

    // Top-left coordinate in input corresponding to this output tile
    int in_row_start = block_out_y * stride - pad_h;
    int in_col_start = block_out_x * stride - pad_w;

    // Allocate dynamically sized shared memory
    extern __shared__ float shared_input[];

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Each thread cooperatively loads part of the shared memory tile.
        // Only use the xy dimensions; z dimension is used for independent output channels.
        int shared_size = sh_width * sh_height;
        int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
        for (int i = tid; i < shared_size; i += (TILE_WIDTH * TILE_HEIGHT)) {
            int sh_y = i / sh_width;
            int sh_x = i % sh_width;
            int in_y = in_row_start + sh_y;
            int in_x = in_col_start + sh_x;
            float val = 0.0f;
            if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                int input_index = b * in_channels * input_height * input_width +
                                  ic * input_height * input_width +
                                  in_y * input_width + in_x;
                val = x[input_index];
            }
            shared_input[i] = val;
        }
        __syncthreads(); // Ensure entire tile is loaded before use

        // Compute local position in shared memory for the output element
        // The shared memory tile corresponds to a region starting at global coordinate (block_out_y*stride - pad_h, block_out_x*stride - pad_w).
        // For an output pixel at (block_out_y + threadIdx.y, block_out_x + threadIdx.x), the corresponding input coordinate is ((block_out_y+threadIdx.y)*stride, (block_out_x+threadIdx.x)*stride).
        // Thus, its location in shared memory is:
        int local_in_y = threadIdx.y * stride + pad_h; 
        int local_in_x = threadIdx.x * stride + pad_w;

        // Accumulate convolution over the kernel window
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int sh_y = local_in_y + kh * dilation_h;
                int sh_x = local_in_x + kw * dilation_w;
                int sh_idx = sh_y * sh_width + sh_x;
                float a = shared_input[sh_idx];

                int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                 ic * kernel_h * kernel_w +
                                 kh * kernel_w + kw;
                float w_val = weight[weight_idx];
                sum += a * w_val;
            }
        }
        // Synchronize only if more input channels remain to avoid overwriting shared memory
        if (ic < in_channels - 1)
            __syncthreads();
    }

    // Write the computed output value
    int out_index = b * out_channels * height_out * width_out +
                    oc * height_out * width_out +
                    out_y * width_out + out_x;
    output[out_index] = sum;
}


// Host function to setup and launch the CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
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
    if (height_out <= 0 || width_out <= 0) return output;

    // Setup block and grid dimensions
    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT, NUM_OC_PER_BLOCK);
    int num_oc_tiles = (out_channels + NUM_OC_PER_BLOCK - 1) / NUM_OC_PER_BLOCK;
    dim3 gridDim(
        (width_out + TILE_WIDTH - 1) / TILE_WIDTH,
        (height_out + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch_size * num_oc_tiles);

    // Compute shared memory size required per block
    int sh_width = TILE_WIDTH * stride + (kernel_w - 1) * dilation_w;
    int sh_height = TILE_HEIGHT * stride + (kernel_h - 1) * dilation_h;
    size_t shared_mem_size = sh_width * sh_height * sizeof(float);

    conv2d_kernel_shared<<<gridDim, blockDim, shared_mem_size>>>(
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
    m.def("forward", &forward, "Conv2D forward (CUDA) with shared memory and minimal synchronizations");
}
