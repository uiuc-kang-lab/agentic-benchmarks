#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions and channel grouping
#define TILE_SIZE 16
#define CHANNELS_PER_BLOCK 4

// Kernel that loads the entire input tile into shared memory once and synchronizes only once
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

    // Compute dimensions for the shared memory tile
    // The tile must cover all input positions needed for a TILE_SIZE x TILE_SIZE output tile
    int shared_height = (TILE_SIZE - 1) * stride + (kernel_h - 1) * dilation_h + 1;
    int shared_width  = (TILE_SIZE - 1) * stride + (kernel_w - 1) * dilation_w + 1;

    // Decode blockIdx.z to retrieve the batch index and output channel tile index
    int num_oc_tiles = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int b  = blockIdx.z / num_oc_tiles;                // batch index
    int oc_tile = blockIdx.z % num_oc_tiles;            // which tile of output channels
    int oc_start = oc_tile * CHANNELS_PER_BLOCK;        

    // Determine output coordinate for this thread
    int tile_row = threadIdx.y;  // in [0, TILE_SIZE)
    int tile_col = threadIdx.x;  
    int out_row = blockIdx.y * TILE_SIZE + tile_row;
    int out_col = blockIdx.x * TILE_SIZE + tile_col;

    // Base input coordinates corresponding to the top-left of the shared tile
    int in_row_start = blockIdx.y * TILE_SIZE * stride - pad_h;
    int in_col_start = blockIdx.x * TILE_SIZE * stride - pad_w;

    // Declare shared memory (dynamically allocated) for the input tile
    // Layout: [in_channels][shared_height][shared_width]
    extern __shared__ float shared_input[];

    // Total number of elements to load into shared memory
    int total_shared_elems = in_channels * shared_height * shared_width;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;

    // Cooperative loading: each thread loads multiple elements
    for (int i = thread_id; i < total_shared_elems; i += block_threads) {
        int ic = i / (shared_height * shared_width);
        int rem = i % (shared_height * shared_width);
        int sh = rem / shared_width;
        int sw = rem % shared_width;

        int global_row = in_row_start + sh;
        int global_col = in_col_start + sw;
        float val = 0.0f;
        if (global_row >= 0 && global_row < input_height && global_col >= 0 && global_col < input_width) {
            int x_index = b * in_channels * input_height * input_width
                        + ic * input_height * input_width
                        + global_row * input_width + global_col;
            val = x[x_index];
        }
        shared_input[i] = val;
    }
    
    // One synchronization after loading shared memory
    __syncthreads();

    // Only compute if the thread's output coordinate is within bounds
    if (out_row < height_out && out_col < width_out) {
        // Accumulate results for up to CHANNELS_PER_BLOCK output channels
        float accum[CHANNELS_PER_BLOCK];
        #pragma unroll
        for (int k = 0; k < CHANNELS_PER_BLOCK; k++) {
            int oc = oc_start + k;
            if (oc < out_channels)
                accum[k] = (bias != nullptr) ? bias[oc] : 0.0f;
            else
                accum[k] = 0.0f;
        }

        // For each input channel, iterate over the kernel window and accumulate
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // Compute shared memory indices corresponding to the current output element
                    int sh = tile_row * stride + kh * dilation_h;
                    int sw = tile_col * stride + kw * dilation_w;
                    float input_val = shared_input[ic * (shared_height * shared_width) + sh * shared_width + sw];

                    #pragma unroll
                    for (int k = 0; k < CHANNELS_PER_BLOCK; k++) {
                        int oc = oc_start + k;
                        if (oc < out_channels) {
                            // Weight layout: [out_channels, in_channels, kernel_h, kernel_w]
                            int weight_index = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                            accum[k] += input_val * weight[weight_index];
                        }
                    }
                }
            }
        }

        // Write the accumulated outputs back to global memory
        for (int k = 0; k < CHANNELS_PER_BLOCK; k++) {
            int oc = oc_start + k;
            if (oc < out_channels) {
                int out_index = b * out_channels * height_out * width_out
                              + oc * height_out * width_out
                              + out_row * width_out + out_col;
                output[out_index] = accum[k];
            }
        }
    }
}


// Forward function called from PyTorch
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

    int batch_size  = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width  = x.size(3);
    
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out  = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    // Grid configuration
    int grid_x = (width_out + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (height_out + TILE_SIZE - 1) / TILE_SIZE;
    int num_oc_tiles = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int grid_z = batch_size * num_oc_tiles;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(grid_x, grid_y, grid_z);

    // Shared memory size: for the input tile for all in_channels
    int shared_height = (TILE_SIZE - 1) * stride + (kernel_h - 1) * dilation_h + 1;
    int shared_width  = (TILE_SIZE - 1) * stride + (kernel_w - 1) * dilation_w + 1;
    size_t shared_mem_size = in_channels * shared_height * shared_width * sizeof(float);

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
    m.def("forward", &forward, "Conv2D forward (CUDA)");
}
