#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block dimensions and channels processed per block in the output channel dimension
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
#define CHANNELS_PER_BLOCK 4

// Kernel leveraging shared memory to cache weight and input tile data
// for reduction of global memory latency.

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

    // Thread indices for spatial tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global output pixel coordinates
    int out_x = blockIdx.x * BLOCK_SIZE_X + tx;
    int out_y = blockIdx.y * BLOCK_SIZE_Y + ty;

    // Determine the output channel tile being processed
    int oc_tile = blockIdx.z;  
    int oc_base = oc_tile * CHANNELS_PER_BLOCK;

    // Return if the output pixel is outside the valid range
    if (out_x >= width_out || out_y >= height_out) return;

    // Compute the top-left corner in the input corresponding to the block's output tile
    int out_tile_x = blockIdx.x * BLOCK_SIZE_X;
    int out_tile_y = blockIdx.y * BLOCK_SIZE_Y;
    int in_tile_x = out_tile_x * stride - pad_w;
    int in_tile_y = out_tile_y * stride - pad_h;

    // Compute shared tile dimensions for the input patch
    // Each output pixel uses a receptive field of size determined by kernel and dilation.
    int tile_width = (BLOCK_SIZE_X - 1) * stride + (kernel_w - 1) * dilation_w + 1;
    int tile_height = (BLOCK_SIZE_Y - 1) * stride + (kernel_h - 1) * dilation_h + 1;

    // Allocate shared memory:
    // We'll reserve space for the weight tile and the input tile.
    // Weight tile size: CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w
    // Input tile size: in_channels * tile_height * tile_width
    extern __shared__ float sdata[];
    int weight_tile_size = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w;
    float* shared_weight = sdata; 
    float* shared_input = sdata + weight_tile_size;

    // Total number of threads in the block (for cooperative loading)
    int block_threads = BLOCK_SIZE_X * BLOCK_SIZE_Y;
    int thread_id = ty * BLOCK_SIZE_X + tx;

    // Load the weight tile (for output channels [oc_base, oc_base+CHANNELS_PER_BLOCK)) into shared memory
    int total_weight_elems = weight_tile_size;
    for (int i = thread_id; i < total_weight_elems; i += block_threads) {
        int oc_offset = i / (in_channels * kernel_h * kernel_w);
        int rem = i % (in_channels * kernel_h * kernel_w);
        int ic = rem / (kernel_h * kernel_w);
        int rem2 = rem % (kernel_h * kernel_w);
        int kh = rem2 / kernel_w;
        int kw = rem2 % kernel_w;
        int global_oc = oc_base + oc_offset;
        if (global_oc < out_channels) {
            int w_idx = global_oc * in_channels * kernel_h * kernel_w +
                        ic * kernel_h * kernel_w +
                        kh * kernel_w + kw;
            shared_weight[i] = weight[w_idx];
        } else {
            shared_weight[i] = 0.0f;
        }
    }
    __syncthreads();

    // Loop over batch dimension
    for (int b = 0; b < batch_size; ++b) {
        // Load the input tile for the current batch element into shared memory
        int tile_elems = in_channels * tile_height * tile_width;
        for (int i = thread_id; i < tile_elems; i += block_threads) {
            int ic = i / (tile_height * tile_width);
            int rem = i % (tile_height * tile_width);
            int iy = rem / tile_width;
            int ix = rem % tile_width;
            int global_y = in_tile_y + iy;
            int global_x = in_tile_x + ix;
            float val = 0.0f;
            if (global_y >= 0 && global_y < input_height && global_x >= 0 && global_x < input_width) {
                int index = b * in_channels * input_height * input_width +
                            ic * input_height * input_width +
                            global_y * input_width + global_x;
                val = x[index];
            }
            shared_input[i] = val;
        }
        __syncthreads();

        // Each thread computes one output pixel at (out_y, out_x) for each output channel in its tile
        float out_vals[CHANNELS_PER_BLOCK];
        for (int k = 0; k < CHANNELS_PER_BLOCK; ++k) {
            int global_oc = oc_base + k;
            out_vals[k] = (global_oc < out_channels && bias != nullptr) ? bias[global_oc] : 0.0f;
        }

        // Determine the starting position inside the input tile for this output pixel
        int in_local_y = ty * stride;
        int in_local_x = tx * stride;

        // Accumulate contributions over all input channels and kernel window
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                int in_y = in_local_y + kh * dilation_h;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_x = in_local_x + kw * dilation_w;
                    int input_idx = ic * (tile_height * tile_width) + iy * tile_width + ix;
                    float in_val = shared_input[input_idx];
                    for (int k = 0; k < CHANNELS_PER_BLOCK; ++k) {
                        int weight_idx = k * (in_channels * kernel_h * kernel_w) +
                                         ic * (kernel_h * kernel_w) +
                                         kh * kernel_w + kw;
                        out_vals[k] += in_val * shared_weight[weight_idx];
                    }
                }
            }
        }

        // Write the computed output values to global memory
        for (int k = 0; k < CHANNELS_PER_BLOCK; ++k) {
            int global_oc = oc_base + k;
            if (global_oc < out_channels) {
                int out_idx = b * out_channels * height_out * width_out +
                              global_oc * height_out * width_out +
                              out_y * width_out + out_x;
                output[out_idx] = out_vals[k];
            }
        }
        __syncthreads(); // Synchronize before next batch iteration
    }
}


// Host forward function

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,  // Optional bias
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

    // Compute shared memory requirements
    int weight_tile_size = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w;
    int tile_height = (BLOCK_SIZE_Y - 1) * stride + (kernel_h - 1) * dilation_h + 1;
    int tile_width = (BLOCK_SIZE_X - 1) * stride + (kernel_w - 1) * dilation_w + 1;
    int input_tile_size = in_channels * tile_height * tile_width;
    int shared_mem_bytes = (weight_tile_size + input_tile_size) * sizeof(float);

    // Set up grid and block dimensions
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(
        (width_out + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (height_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK
    );

    conv2d_kernel_shared<<<blocks, threads, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Conv2D forward with shared memory optimizations (CUDA)");
}
