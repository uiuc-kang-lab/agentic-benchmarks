#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Macros for input checking
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tile dimensions for output block
#define TILE_H 16
#define TILE_W 16

// CUDA kernel leveraging shared memory to reduce global memory latency.
// Each block computes a tile of the output for one batch sample and one output channel.
// Within the block, threads cooperatively load a tile of the corresponding input channel into shared memory.
// The kernel then computes the convolution using the cached data and proper synchronization to avoid race conditions.

__global__ void conv2d_shared_mem_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    // Determine indices based on grid and block dimensions
    // gridDim.x: batch, gridDim.y: output channel, gridDim.z: tiled spatial blocks
    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;

    // Compute number of tiles in x direction
    int num_tiles_x = (out_width + TILE_W - 1) / TILE_W;
    int tile_idx = blockIdx.z;  // tile index in the 2D tiling
    int tile_row = tile_idx / num_tiles_x;
    int tile_col = tile_idx % num_tiles_x;

    // Top-left corner of the output tile
    int out_start_y = tile_row * TILE_H;
    int out_start_x = tile_col * TILE_W;

    // Each thread computes one output element within the tile
    int local_y = threadIdx.y;
    int local_x = threadIdx.x;
    int out_y = out_start_y + local_y;
    int out_x = out_start_x + local_x;

    // Compute the dimensions of the shared memory tile required
    // Shared tile height: (TILE_H - 1)*stride + (kernel_size - 1)*dilation + 1
    // Shared tile width:  (TILE_W - 1)*stride + (kernel_size - 1)*dilation + 1
    int sh_height = (TILE_H - 1) * stride + (kernel_size - 1) * dilation + 1;
    int sh_width  = (TILE_W - 1) * stride + (kernel_size - 1) * dilation + 1;

    // The input tile (to be loaded in shared memory) starts at:
    int in_tile_y = out_start_y * stride - padding;
    int in_tile_x = out_start_x * stride - padding;

    float sum = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Declare dynamically allocated shared memory
        extern __shared__ float shmem[];  // size should be sh_height * sh_width floats

        // Cooperatively load the required input tile for current input channel into shared memory
        int total_sh_elems = sh_height * sh_width;
        int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        for (int idx = thread_id; idx < total_sh_elems; idx += blockDim.x * blockDim.y) {
            int sh_y = idx / sh_width;
            int sh_x = idx % sh_width;
            int in_y = in_tile_y + sh_y;
            int in_x = in_tile_x + sh_x;
            float val = 0.0f;
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int input_idx = batch_idx * (in_channels * in_height * in_width) +
                                ic * (in_height * in_width) +
                                in_y * in_width + in_x;
                val = input[input_idx];
            }
            shmem[sh_y * sh_width + sh_x] = val;
        }
        __syncthreads();

        // Each thread computes the convolution for its output pixel using the shared memory data
        if (out_y < out_height && out_x < out_width) {
            float accum = 0.0f;
            // For each element in the kernel window
            #pragma unroll
            for (int ky = 0; ky < kernel_size; ky++) {
                #pragma unroll
                for (int kx = 0; kx < kernel_size; kx++) {
                    int sh_y = local_y * stride + ky * dilation;
                    int sh_x = local_x * stride + kx * dilation;
                    float in_val = shmem[sh_y * sh_width + sh_x];
                    int weight_idx = ((out_channel * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                    accum += in_val * weight[weight_idx];
                }
            }
            sum += accum;
        }
        __syncthreads(); // Ensure all threads have finished using shared memory before next channel
    }

    // Write the final result to the output tensor if within bounds
    if (out_y < out_height && out_x < out_width) {
        if (bias != nullptr) {
            sum += bias[out_channel];
        }
        int out_idx = batch_idx * (out_channels * out_height * out_width) +
                      out_channel * (out_height * out_width) +
                      out_y * out_width + out_x;
        output[out_idx] = sum;
    }
}


// Host function to prepare and launch the kernel
// Falls back to torch::conv2d if groups != 1

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

    TORCH_CHECK(groups == 1, "Only groups==1 is supported in conv2d_shared_mem_kernel");

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);

    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);  // assuming square kernel

    // Compute output dimensions
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    // Calculate tiling dimensions for output
    int num_tiles_x = (out_width + TILE_W - 1) / TILE_W;
    int num_tiles_y = (out_height + TILE_H - 1) / TILE_H;
    int total_tiles = num_tiles_x * num_tiles_y;

    // Set up grid and block dimensions
    dim3 grid(batch, out_channels, total_tiles);
    dim3 block(TILE_W, TILE_H);

    // Compute shared memory size (in bytes) needed per block
    int sh_height = (TILE_H - 1) * stride + (kernel_size - 1) * dilation + 1;
    int sh_width  = (TILE_W - 1) * stride + (kernel_size - 1) * dilation + 1;
    size_t shared_mem_size = sh_height * sh_width * sizeof(float);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv2d_shared_mem_kernel<<<grid, block, shared_mem_size>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Convolution forward using shared memory to reduce global memory latency");
}
