#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tile dimensions for spatial output and input channel chunk
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define TILE_C 8  // number of input channels processed per chunk

// This kernel is designed for groups==1, dilation==1 and stride==1
// It leverages shared memory for both weight and input tile caching to reduce global memory latency.
// Each block computes a TILE_HEIGHT x TILE_WIDTH patch for one output channel of one sample.

__global__ void conv2d_shared_tile_kernel(
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
    int padding) {

    // Determine sample and output channel
    int oc = blockIdx.z % out_channels;
    int n  = blockIdx.z / out_channels;

    // Compute output coordinates
    int out_y = blockIdx.y * TILE_HEIGHT + threadIdx.y; 
    int out_x = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float acc = 0.0f;

    // Declare dynamic shared memory. We'll use it for loading a chunk of weight and input tile.
    extern __shared__ float sh_mem[];
    // Allocate shared memory for weight chunk: fixed size: TILE_C * kernel_size * kernel_size
    float* sh_weight = sh_mem;
    // Followed by shared memory for input tile chunk: TILE_C * (TILE_HEIGHT + kernel_size - 1) * (TILE_WIDTH + kernel_size - 1)
    float* sh_input = sh_mem + (TILE_C * kernel_size * kernel_size);

    // Constants for input tile dimensions in shared memory
    const int SH_TILE_HEIGHT = TILE_HEIGHT + kernel_size - 1;
    const int SH_TILE_WIDTH = TILE_WIDTH + kernel_size - 1;

    // Loop over input channels in chunks of TILE_C
    for (int c0 = 0; c0 < in_channels; c0 += TILE_C) {
        int current_tile_channels = ((in_channels - c0) < TILE_C) ? (in_channels - c0) : TILE_C;

        // Load weight chunk for current input channels
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int weight_tile_size = current_tile_channels * kernel_size * kernel_size;
        for (int i = tid; i < weight_tile_size; i += blockDim.x * blockDim.y) {
            int local_c = i / (kernel_size * kernel_size);
            int rem = i % (kernel_size * kernel_size);
            int k_row = rem / kernel_size;
            int k_col = rem % kernel_size;
            int global_c = c0 + local_c;  // global input channel
            // Weight layout: [out_channels, in_channels, kernel_size, kernel_size]
            int weight_idx = oc * (in_channels * kernel_size * kernel_size) + global_c * (kernel_size * kernel_size) + k_row * kernel_size + k_col;
            sh_weight[i] = weight[weight_idx];
        }
        __syncthreads();

        // Load input tile chunk for current input channels into shared memory
        int input_tile_elems = current_tile_channels * SH_TILE_HEIGHT * SH_TILE_WIDTH;
        for (int i = tid; i < input_tile_elems; i += blockDim.x * blockDim.y) {
            int local_channel = i / (SH_TILE_HEIGHT * SH_TILE_WIDTH);
            int rem = i % (SH_TILE_HEIGHT * SH_TILE_WIDTH);
            int tile_row = rem / SH_TILE_WIDTH;
            int tile_col = rem % SH_TILE_WIDTH;
            // Compute the top-left corner of the input tile for this block
            int base_y = blockIdx.y * TILE_HEIGHT - padding;
            int base_x = blockIdx.x * TILE_WIDTH - padding;
            int in_y = base_y + tile_row;
            int in_x = base_x + tile_col;
            int global_channel = c0 + local_channel;
            float in_val = 0.0f;
            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                int input_idx = n * (in_channels * in_h * in_w) + global_channel * (in_h * in_w) + in_y * in_w + in_x;
                in_val = input[input_idx];
            }
            int sh_index = local_channel * (SH_TILE_HEIGHT * SH_TILE_WIDTH) + tile_row * SH_TILE_WIDTH + tile_col;
            sh_input[sh_index] = in_val;
        }
        __syncthreads();

        // Only compute if the output index is within range
        if (out_y < out_h && out_x < out_w) {
            // For each channel in the current chunk
            for (int local_c = 0; local_c < current_tile_channels; local_c++) {
                // Iterate over the kernel window
                for (int k_row = 0; k_row < kernel_size; k_row++) {
                    for (int k_col = 0; k_col < kernel_size; k_col++) {
                        // The corresponding input location in shared memory
                        int sh_input_index = local_c * (SH_TILE_HEIGHT * SH_TILE_WIDTH) + (threadIdx.y + k_row) * SH_TILE_WIDTH + (threadIdx.x + k_col);
                        float in_val = sh_input[sh_input_index];
                        int sh_weight_index = local_c * (kernel_size * kernel_size) + k_row * kernel_size + k_col;
                        float w_val = sh_weight[sh_weight_index];
                        acc += in_val * w_val;
                    }
                }
            }
        }
        __syncthreads();
    } // end for each input channel chunk

    // Write the computed output if within bounds, adding bias if provided
    if (out_y < out_h && out_x < out_w) {
        if (bias != nullptr) {
            acc += bias[oc];
        }
        int out_idx = n * (out_channels * out_h * out_w) + oc * (out_h * out_w) + out_y * out_w + out_x;
        output[out_idx] = acc;
    }
}

// Host forward function
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

    // This optimized kernel supports only groups==1, dilation==1 and stride==1
    if (groups != 1 || dilation != 1 || stride != 1) {
        if (bias.has_value()) {
            return torch::conv2d(x, weight, bias.value(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        } else {
            return torch::conv2d(x, weight, torch::Tensor(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        }
    }

    // Get input dimensions: x is [batch_size, in_channels, in_h, in_w]
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    // Weight layout: [out_channels, in_channels, kernel_size, kernel_size]
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // assume square kernel

    // Compute output dimensions (assumes stride==1)
    int out_h = (in_h + 2 * padding - kernel_size) + 1;
    int out_w = (in_w + 2 * padding - kernel_size) + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    // Define block and grid dimensions
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH,
              (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
              batch_size * out_channels);

    // Calculate shared memory size in bytes
    // Shared memory per block: for weight tile (TILE_C * kernel_size * kernel_size) and
    // for input tile chunk (TILE_C * (TILE_HEIGHT + kernel_size - 1) * (TILE_WIDTH + kernel_size - 1))
    size_t shared_mem_size = (TILE_C * kernel_size * kernel_size +
                              TILE_C * (TILE_HEIGHT + kernel_size - 1) * (TILE_WIDTH + kernel_size - 1))
                              * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv2d_shared_tile_kernel<<<grid, block, shared_mem_size, stream>>>(
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
        padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution using tiled shared memory");
}
