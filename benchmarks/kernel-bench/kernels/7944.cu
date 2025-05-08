#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define tile dimensions for output per block
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// This kernel uses shared memory to load the required input patch for each output tile and then
// performs the convolution for one output channel. It supports stride, padding, and dilation
// (with the assumption of a square input and square kernel) for groups==1.

__global__ void conv_shared_mem_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int padding,
    int stride,
    int dilation) {

    // Determine which batch and output channel this block is processing
    int b = blockIdx.z / out_channels;
    int oc = blockIdx.z % out_channels;

    // Determine the spatial tile coordinates for the output
    int out_tile_y = blockIdx.x * TILE_HEIGHT;
    int out_tile_x = blockIdx.y * TILE_WIDTH;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int out_y = out_tile_y + ty;
    int out_x = out_tile_x + tx;

    // Calculate the dimensions of the shared memory tile based on stride and dilation
    int shared_width  = (TILE_WIDTH - 1) * stride + (kernel_size - 1) * dilation + 1;
    int shared_height = (TILE_HEIGHT - 1) * stride + (kernel_size - 1) * dilation + 1;

    // Declare shared memory buffer
    extern __shared__ float tile[];

    float sum = 0.0f;

    // The origin in the input corresponding to the top-left of the shared memory tile
    int in_tile_origin_y = out_tile_y * stride - padding;
    int in_tile_origin_x = out_tile_x * stride - padding;

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Load the shared memory tile for this input channel
        // Each thread loads one or more elements from global memory into shared memory.
        for (int i = ty; i < shared_height; i += blockDim.y) {
            for (int j = tx; j < shared_width; j += blockDim.x) {
                int in_y = in_tile_origin_y + i;
                int in_x = in_tile_origin_x + j;
                float val = 0.0f;
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = ((b * in_channels + ic) * in_height + in_y) * in_width + in_x;
                    val = input[input_idx];
                }
                tile[i * shared_width + j] = val;
            }
        }
        __syncthreads();

        // Only threads within the output bounds compute convolution
        if (out_y < out_height && out_x < out_width) {
            // The top-left corner in shared memory corresponding to this output pixel
            int tile_y = ty * stride;
            int tile_x = tx * stride;
            
            // Iterate over the kernel window
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int idx_y = tile_y + ky * dilation;
                    int idx_x = tile_x + kx * dilation;
                    float in_val = tile[idx_y * shared_width + idx_x];
                    // Weight indexing: [oc, ic, ky, kx]
                    int weight_idx = (((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx);
                    float w = weight[weight_idx];
                    sum += in_val * w;
                }
            }
        }
        __syncthreads();
    }

    // Write the computed sum to the output tensor if within valid bounds
    if (out_y < out_height && out_x < out_width) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int output_idx = ((b * out_channels + oc) * out_height + out_y) * out_width + out_x;
        output[output_idx] = sum;
    }
}

// Host function that wraps the custom CUDA kernel and interfaces with PyTorch
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    TORCH_CHECK(groups == 1, "Only groups==1 is supported in this custom kernel");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // Assumes square kernel

    // Compute output dimensions using the standard convolution formula
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch, out_channels, out_height, out_width}, input.options());

    // Define block and grid dimensions
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid(
        (out_height + TILE_HEIGHT - 1) / TILE_HEIGHT,
        (out_width  + TILE_WIDTH  - 1) / TILE_WIDTH,
        batch * out_channels);
              
    // Calculate the required shared memory size
    int shared_width  = (TILE_WIDTH - 1) * stride + (kernel_size - 1) * dilation + 1;
    int shared_height = (TILE_HEIGHT - 1) * stride + (kernel_size - 1) * dilation + 1;
    size_t shared_mem_size = shared_width * shared_height * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const float* input_ptr  = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr   = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr       = output.data_ptr<float>();

    conv_shared_mem_kernel<<<grid, block, shared_mem_size, stream>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        padding,
        stride,
        dilation);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution using shared memory");
}
