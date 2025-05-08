#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macros for checking tensor properties
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel implements 2D convolution for groups==1 and dilation==1 using a two-level optimization:
// 1. Input tiling: Each thread block loads a tile (with halo) of the input image into shared memory to
//    reuse overlapping data needed for adjacent output pixels.
// 2. Weight caching via __ldg: Instead of loading the weight into shared memory with synchronization overhead,
//    we use the read-only data cache (__ldg) to fetch weight values.

// Block configuration: Each block is assigned a tile of the output for a given sample and output channel.
// The grid dimension in z encodes (batch_index, output_channel).

__global__ void conv2d_tiled_ldg_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_size,
    int out_h,
    int out_w,
    int stride,
    int padding) {

    // Determine the output channel and batch index
    int oc = blockIdx.z % out_channels;
    int n  = blockIdx.z / out_channels;

    // Block dimensions
    const int Bx = blockDim.x;  // number of threads in x (output tile width)
    const int By = blockDim.y;  // number of threads in y (output tile height)

    // Compute the output coordinates for this thread
    int out_x = blockIdx.x * Bx + threadIdx.x;
    int out_y = blockIdx.y * By + threadIdx.y;

    // Return if outside the output
    bool valid = (out_x < out_w && out_y < out_h);

    // Compute the top-left coordinate in the input corresponding to the top-left output of this block
    // Each block covers a tile of output starting at:
    //   out_start_x = blockIdx.x * Bx
    //   out_start_y = blockIdx.y * By
    // Thus, the corresponding input starting coordinate (accounting for stride and padding) is:
    //   in_tile_start_x = out_start_x * stride - padding
    //   in_tile_start_y = out_start_y * stride - padding
    int out_tile_start_x = blockIdx.x * Bx;
    int out_tile_start_y = blockIdx.y * By;
    int in_tile_start_x = out_tile_start_x * stride - padding;
    int in_tile_start_y = out_tile_start_y * stride - padding;

    // Determine the dimensions of the input tile required in shared memory.
    // For an output tile of size (Bx, By) and stride, we need a tile of size:
    //   tile_w = (Bx - 1) * stride + kernel_size
    //   tile_h = (By - 1) * stride + kernel_size
    int tile_w = (Bx - 1) * stride + kernel_size;
    int tile_h = (By - 1) * stride + kernel_size;

    // Allocate shared memory for the input tile (one channel at a time)
    extern __shared__ float shared_input[];  // size: tile_h * tile_w floats

    float sum = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // --- Load input tile for channel 'ic' into shared memory ---
        // The tile covers indices [0, tile_h) x [0, tile_w) in shared memory.
        int num_tile_elements = tile_h * tile_w;
        int thread_linear = threadIdx.y * Bx + threadIdx.x;
        
        // Each thread loads one or more elements of the tile
        for (int idx = thread_linear; idx < num_tile_elements; idx += Bx * By) {
            int tile_i = idx / tile_w; // row within the tile
            int tile_j = idx % tile_w; // col within the tile
            int global_row = in_tile_start_y + tile_i;  // corresponding row in input image
            int global_col = in_tile_start_x + tile_j;  // corresponding col in input image
            float in_val = 0.0f;
            if (global_row >= 0 && global_row < in_h && global_col >= 0 && global_col < in_w) {
                int input_idx = n * (in_channels * in_h * in_w) +
                                ic * (in_h * in_w) +
                                global_row * in_w + global_col;
                in_val = input[input_idx];
            }
            shared_input[idx] = in_val;
        }
        __syncthreads();

        // --- Compute convolution for this channel using the shared input tile ---
        // The output pixel computed by this thread corresponds to a window in the shared tile.
        // Its top-left corner in the tile is:
        int local_x = threadIdx.x * stride;  // offset within the shared tile
        int local_y = threadIdx.y * stride;
        
        // Loop over the filter kernel for this input channel
        for (int ki = 0; ki < kernel_size; ++ki) {
            for (int kj = 0; kj < kernel_size; ++kj) {
                // read the corresponding input value from shared memory
                int tile_r = local_y + ki;
                int tile_c = local_x + kj;
                float in_val = shared_input[tile_r * tile_w + tile_c];
                
                // Compute the weight index and load via __ldg (from the read-only cache)
                int weight_idx = oc * (in_channels * kernel_size * kernel_size) +
                                 ic * (kernel_size * kernel_size) +
                                 ki * kernel_size + kj;
                float w_val = __ldg(&weight[weight_idx]);
                sum += in_val * w_val;
            }
        }
        __syncthreads();  // prepare for next input channel tile loading
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // Write the computed sum to the output tensor
    int output_idx = n * (out_channels * out_h * out_w) +
                     oc * (out_h * out_w) +
                     out_y * out_w + out_x;
    output[output_idx] = sum;
}


// Host function for the forward pass
// Supports only groups == 1 and dilation == 1, otherwise falls back to torch::conv2d

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    // This kernel supports only groups == 1 and dilation == 1
    if (groups != 1 || dilation != 1) {
        if (bias.has_value()) {
            return torch::conv2d(x, weight, bias.value(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        } else {
            return torch::conv2d(x, weight, torch::Tensor(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        }
    }

    // Get input dimensions
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    // Weight dimensions: [out_channels, in_channels, kernel_size, kernel_size]
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);  // assume square kernels

    // Compute output dimensions
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    // Allocate output tensor
    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    // Define block and grid dimensions
    // Using a 2D block for the output tile
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH,
              (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
              batch_size * out_channels);

    // Compute shared memory size for each block
    // tile_w = (TILE_WIDTH - 1) * stride + kernel_size
    // tile_h = (TILE_HEIGHT - 1) * stride + kernel_size
    int tile_w = (TILE_WIDTH - 1) * stride + kernel_size;
    int tile_h = (TILE_HEIGHT - 1) * stride + kernel_size;
    size_t shared_mem_size = tile_w * tile_h * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv2d_tiled_ldg_kernel<<<grid, block, shared_mem_size, stream>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_size,
        out_h,
        out_w,
        stride,
        padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution combining input tiling with __ldg weight access");
}
