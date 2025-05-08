#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses shared memory tiling to reduce redundant global memory accesses
// and improves coalescing by loading extended input tiles that are reused by threads
// to compute convolution outputs. It also supports grouped convolution and optional bias.

// Note: For simplicity, we assume the kernel is launched with 2D thread blocks and 1D grid in z for output channels.

__global__ void conv2d_shared_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_height,
    int kernel_width,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Determine output coordinates and output channel from block and thread indices
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;  // Each block in z corresponds to an output channel

    if (out_x >= out_width || out_y >= out_height || oc >= out_channels) return;

    // Compute group information
    int group_out_channels = out_channels / groups;
    int group = oc / group_out_channels;
    int in_channels_per_group = in_channels / groups;

    // Shared memory dimensions for the input tile
    // Each block processes a tile of output of size (blockDim.y x blockDim.x), but to compute convolution,
    // we need a halo around the tile:
    int shared_tile_width = blockDim.x * stride + (kernel_width - 1) * dilation;
    int shared_tile_height = blockDim.y * stride + (kernel_height - 1) * dilation;

    // Declare dynamically allocated shared memory
    extern __shared__ float shmem[]; // size: shared_tile_width * shared_tile_height

    // Loop over batch dimension
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;

        // Loop over input channels corresponding to the group
        for (int c = 0; c < in_channels_per_group; ++c) {
            int input_channel = group * in_channels_per_group + c;

            // Compute top-left indices in input for this block's tile
            // These are the coordinates corresponding to the shared memory tile
            int tile_origin_x = blockIdx.x * blockDim.x * stride - padding;
            int tile_origin_y = blockIdx.y * blockDim.y * stride - padding;

            // Load the input tile for this channel into shared memory.
            // Each thread cooperatively loads multiple elements from global memory.
            for (int i = threadIdx.y; i < shared_tile_height; i += blockDim.y) {
                for (int j = threadIdx.x; j < shared_tile_width; j += blockDim.x) {
                    int in_x = tile_origin_x + j;
                    int in_y = tile_origin_y + i;
                    float val = 0.0f;
                    if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                        int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                        val = input[input_idx];
                    }
                    shmem[i * shared_tile_width + j] = val;
                }
            }
            __syncthreads();

            // Each thread now computes its output using the shared memory tile.
            // The location in shared memory corresponding to the top-left of the receptive field for this output pixel:
            int local_x = threadIdx.x * stride;
            int local_y = threadIdx.y * stride;

            // Accumulate over the kernel window
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int shmem_x = local_x + kw * dilation;
                    int shmem_y = local_y + kh * dilation;
                    float input_val = shmem[shmem_y * shared_tile_width + shmem_x];
                    // Weight index: [oc, c, kh, kw]
                    int weight_idx = ((oc * in_channels_per_group + c) * kernel_height + kh) * kernel_width + kw;
                    float weight_val = weight[weight_idx];
                    sum += input_val * weight_val;
                }
            }

            __syncthreads(); // Ensure shared memory is ready for next channel load
        }

        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[oc];
        }

        // Write the result to the output tensor for the current batch element
        int output_idx = ((b * out_channels + oc) * out_height + out_y) * out_width + out_x;
        output[output_idx] = sum;
    }
}

// The forward function sets up dimensions, computes shared memory size, and launches the kernel.

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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    // Compute output spatial dimensions
    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = x.options();
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    // Define block size (tile size in output space)
    dim3 block_size(16, 16);
    dim3 grid_size((out_width + block_size.x - 1) / block_size.x,
                   (out_height + block_size.y - 1) / block_size.y,
                   out_channels);

    // Compute shared memory size per block
    int shared_tile_width = block_size.x * stride + (kernel_width - 1) * dilation;
    int shared_tile_height = block_size.y * stride + (kernel_height - 1) * dilation;
    size_t shared_mem_size = shared_tile_width * shared_tile_height * sizeof(float);

    conv2d_shared_tiled_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride,
        padding,
        dilation,
        groups);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Shared Memory Tiling");
}
