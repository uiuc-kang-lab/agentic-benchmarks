#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel using shared memory tiling for small input sizes.
// This kernel uses a 3D grid where grid.z combines the batch and output channel dimensions
// and a 2D block to cover a tile of the convolution output. It loads the required input patch
// into shared memory once per input channel to reduce global memory accesses.

__global__ void conv2d_optimized_kernel(
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

    // Combine batch and out_channel from grid.z
    int idx = blockIdx.z;
    int n = idx / out_channels;
    int oc = idx % out_channels;

    // Compute output pixel location from 2D block indices
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Determine the size of the shared memory tile.
    // For a tile of output of size (blockDim.x, blockDim.y), we need a patch from input of size:
    // tile_width  = blockDim.x * stride + (kernel_size - 1) * dilation
    // tile_height = blockDim.y * stride + (kernel_size - 1) * dilation
    int tile_width  = blockDim.x * stride + (kernel_size - 1) * dilation;
    int tile_height = blockDim.y * stride + (kernel_size - 1) * dilation;

    // Compute the top-left coordinate of the input patch corresponding to the output tile.
    int out_tile_origin_x = blockIdx.x * blockDim.x;
    int out_tile_origin_y = blockIdx.y * blockDim.y;
    int in_origin_x = out_tile_origin_x * stride - padding;
    int in_origin_y = out_tile_origin_y * stride - padding;

    extern __shared__ float shared_tile[]; // Shared memory for one input channel tile

    float sum = 0.0f;

    // Loop over each input channel
    for (int ic = 0; ic < in_channels; ic++) {
        // Cooperative loading of the input patch for the current input channel into shared memory
        int num_tile_elements = tile_width * tile_height;
        int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = thread_id; i < num_tile_elements; i += blockDim.x * blockDim.y) {
            int ty = i / tile_width; 
            int tx = i % tile_width;
            int global_x = in_origin_x + tx;
            int global_y = in_origin_y + ty;
            if (global_x >= 0 && global_x < in_width && global_y >= 0 && global_y < in_height) {
                shared_tile[i] = input[n * in_channels * in_height * in_width +
                                           ic * in_height * in_width +
                                           global_y * in_width + global_x];
            } else {
                shared_tile[i] = 0.0f;
            }
        }
        __syncthreads();

        // Only perform computation if the output coordinate is within bounds
        if (out_x < out_width && out_y < out_height) {
            // For the current thread's output pixel, its corresponding top-left location in shared memory
            int shared_offset_x = (threadIdx.x * stride);
            int shared_offset_y = (threadIdx.y * stride);
            
            // Accumulate over the convolution kernel window
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int shared_x = shared_offset_x + kx * dilation;
                    int shared_y = shared_offset_y + ky * dilation;
                    float in_val = shared_tile[shared_y * tile_width + shared_x];
                    float w = weight[oc * in_channels * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size +
                                     ky * kernel_size + kx];
                    sum += in_val * w;
                }
            }
        }
        __syncthreads(); // Prepare shared memory for next input channel
    }

    // Write the output if the thread actually corresponds to a valid output pixel
    if (out_x < out_width && out_y < out_height) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[n * out_channels * out_height * out_width +
               oc * out_height * out_width +
               out_y * out_width + out_x] = sum;
    }
}

// Combined forward function that selects the optimized kernel for small inputs (and groups==1), and falls back on
// the standard torch::conv2d call for larger inputs or grouped convolution.

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

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width  = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Use optimized kernel only for small images and non-grouped convolutions
    if (in_height <= 32 && in_width <= 32 && groups == 1) {
        auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

        // Choose block dimensions. Here we select a modest block size to balance work and shared memory usage.
        const int THREADS_X = 8;
        const int THREADS_Y = 8;
        dim3 threads(THREADS_X, THREADS_Y);
        // Grid: spatial tiling in x and y, and combine batch and output channels in z
        dim3 blocks(
            (out_width  + THREADS_X - 1) / THREADS_X,
            (out_height + THREADS_Y - 1) / THREADS_Y,
            batch * out_channels);

        // Shared memory tile dimensions
        int tile_width  = THREADS_X * stride + (kernel_size - 1) * dilation;
        int tile_height = THREADS_Y * stride + (kernel_size - 1) * dilation;
        size_t shared_memory_size = tile_width * tile_height * sizeof(float);

        conv2d_optimized_kernel<<<blocks, threads, shared_memory_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch, in_channels, out_channels,
            in_height, in_width,
            out_height, out_width,
            kernel_size, stride, padding, dilation);

        return output;
    } else {
        // Fallback to highly optimized PyTorch library call for larger tensors or grouped convolutions
        if (bias.has_value()) {
            return torch::conv2d(x, weight, bias.value(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        } else {
            return torch::conv2d(x, weight, torch::Tensor(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive CUDA 2D Convolution with Shared Memory Tiling");
}
