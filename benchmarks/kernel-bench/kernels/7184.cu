#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This CUDA kernel uses shared memory to cache input tiles that are frequently reused in 2D convolution.
// Each block computes a tile of outputs for a given batch and output channel. The input tile (with halo) is loaded
// into shared memory, reducing global memory latency. __syncthreads() is used to ensure proper synchronization
// and avoid race conditions.

__global__ void conv2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Define tile dimensions for output computed per block
    const int TILE_WIDTH = 16;
    // Effective kernel size considering dilation
    int effective_kernel = dilation * (kernel_size - 1) + 1;
    // Shared memory tile size: each block loads a tile of input data of size "tile_size x tile_size"
    // that covers the output tile plus the halo required for the convolution.
    int tile_size = (TILE_WIDTH - 1) * stride + effective_kernel;

    // Determine indices for the current block
    int n = blockIdx.x;       // batch index
    int oc = blockIdx.y;      // output channel index

    // Calculate tiling in the output spatial domain
    int grid_x = (out_width + TILE_WIDTH - 1) / TILE_WIDTH;
    int tile_idx = blockIdx.z;
    int tile_row = tile_idx / grid_x;
    int tile_col = tile_idx % grid_x;

    // Origin for the output tile
    int out_tile_origin_row = tile_row * TILE_WIDTH;
    int out_tile_origin_col = tile_col * TILE_WIDTH;
    
    // Corresponding origin in the input (accounting for stride and padding)
    int in_tile_origin_row = out_tile_origin_row * stride - padding;
    int in_tile_origin_col = out_tile_origin_col * stride - padding;

    // Thread indices within the block
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Global output coordinates
    int out_row = out_tile_origin_row + ty;
    int out_col = out_tile_origin_col + tx;

    float sum = 0.0f;

    // Allocate shared memory for the input tile. The size is tile_size x tile_size (in floats).
    extern __shared__ float sdata[];

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Load the shared memory tile from global memory
        int total_elems = tile_size * tile_size;
        int thread_id = ty * TILE_WIDTH + tx;
        for (int i = thread_id; i < total_elems; i += TILE_WIDTH * TILE_WIDTH) {
            int sh_row = i / tile_size;
            int sh_col = i % tile_size;
            int in_row = in_tile_origin_row + sh_row;
            int in_col = in_tile_origin_col + sh_col;
            float val = 0.0f;
            if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width) {
                int input_idx = n * in_channels * in_height * in_width +
                                ic * in_height * in_width +
                                in_row * in_width +
                                in_col;
                val = input[input_idx];
            }
            sdata[i] = val;
        }
        __syncthreads();

        // Each thread computes its output using the shared memory tile for this input channel
        if (out_row < out_height && out_col < out_width) {
            // The top-left of the receptive field in shared memory for this output
            int sh_origin_row = ty * stride;
            int sh_origin_col = tx * stride;
            float partial = 0.0f;
            #pragma unroll
            for (int ky = 0; ky < kernel_size; ky++) {
                #pragma unroll
                for (int kx = 0; kx < kernel_size; kx++) {
                    int sh_r = sh_origin_row + ky * dilation;
                    int sh_c = sh_origin_col + kx * dilation;
                    float in_val = sdata[sh_r * tile_size + sh_c];
                    int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size +
                                     ky * kernel_size +
                                     kx;
                    partial += in_val * weight[weight_idx];
                }
            }
            sum += partial;
        }
        __syncthreads();
    }

    // Write the computed output value back to global memory if within bounds
    if (out_row < out_height && out_col < out_width) {
        int out_idx = n * out_channels * out_height * out_width +
                      oc * out_height * out_width +
                      out_row * out_width +
                      out_col;
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[out_idx] = sum;
    }
}


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

    TORCH_CHECK(groups == 1, "Only groups==1 is supported");

    // Extract tensor dimensions
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2); // Square kernel assumed

    // Compute output dimensions
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    // Define block and grid dimensions
    const int TILE_WIDTH = 16;
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    int grid_x = (out_width + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (out_height + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 blocks(batch, out_channels, grid_x * grid_y);

    // Compute shared memory size per block (tile_size x tile_size floats)
    int effective_kernel = dilation * (kernel_size - 1) + 1;
    int tile_size = (TILE_WIDTH - 1) * stride + effective_kernel;
    size_t shared_mem_size = tile_size * tile_size * sizeof(float);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv2d_shared_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &forward, "2D Convolution using shared memory optimization");
}
