#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized 2D convolution kernel using tiling and shared memory for improved data reuse and coalesced global memory access.
__global__ void tiled_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr
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

    // Declare shared memory buffer for the input tile
    extern __shared__ float smem[];

    // Each block processes a tile of the output. The block dimensions define the tile size.
    const int tile_w = blockDim.x;  // e.g. 16
    const int tile_h = blockDim.y;  // e.g. 16

    // Compute the dimensions of the shared memory tile
    // For each output tile, we need extra border pixels from the input:
    // shared_width = tile_w * stride + (kernel_width - 1) * dilation
    // shared_height = tile_h * stride + (kernel_height - 1) * dilation
    const int shared_width = tile_w * stride + (kernel_width - 1) * dilation;
    const int shared_height = tile_h * stride + (kernel_height - 1) * dilation;

    // Global output coordinates
    int out_x = blockIdx.x * tile_w + threadIdx.x;
    int out_y = blockIdx.y * tile_h + threadIdx.y;
    int oc = blockIdx.z;  // each block in z-dimension corresponds to an output channel

    if (out_x < out_width && out_y < out_height && oc < out_channels) {
        // Loop over each element in the batch
        for (int b = 0; b < batch_size; ++b) {
            float sum = 0.0f;
            
            // Determine the group and input channels per group
            int group_out_channels = out_channels / groups;
            int group = oc / group_out_channels;
            int in_channels_per_group = in_channels / groups;

            // Loop over each input channel within the group
            for (int c = 0; c < in_channels_per_group; ++c) {
                int input_channel = group * in_channels_per_group + c;

                // Calculate the top-left coordinate of the input tile for this block
                int tile_origin_x = blockIdx.x * tile_w * stride - padding;
                int tile_origin_y = blockIdx.y * tile_h * stride - padding;

                // Total number of elements in the shared memory tile
                int num_shared = shared_width * shared_height;
                int threads_per_block = tile_w * tile_h;
                int tid = threadIdx.y * tile_w + threadIdx.x;

                // Load the input tile from global memory into shared memory cooperatively
                for (int i = tid; i < num_shared; i += threads_per_block) {
                    int sx = i % shared_width;
                    int sy = i / shared_width;
                    int in_x = tile_origin_x + sx;
                    int in_y = tile_origin_y + sy;
                    float val = 0.0f;
                    if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                        int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                        val = input[input_idx];
                    }
                    smem[i] = val;
                }
                __syncthreads();

                // Each thread computes its corresponding output element using the loaded shared memory tile
                // The starting coordinate in the shared memory tile for this output is:
                int local_x = threadIdx.x * stride;
                int local_y = threadIdx.y * stride;
                
                // Accumulate the convolution sum over the kernel window
                for (int kh = 0; kh < kernel_height; ++kh) {
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        int sx = local_x + kw * dilation;
                        int sy = local_y + kh * dilation;
                        if (sx < shared_width && sy < shared_height) {
                            float in_val = smem[sy * shared_width + sx];
                            int weight_idx = (((oc * in_channels_per_group + c) * kernel_height + kh) * kernel_width) + kw;
                            float w_val = weight[weight_idx];
                            sum += in_val * w_val;
                        }
                    }
                }
                __syncthreads(); // Ensure shared memory is free before next channel load
            }
            
            // Apply bias if provided
            if (bias != nullptr) {
                sum += bias[oc];
            }
            
            // Write the computed value to the output tensor
            int output_idx = ((b * out_channels + oc) * out_height + out_y) * out_width + out_x;
            output[output_idx] = sum;
        }
    }
}

// Forward function sets up the kernel launch. It integrates the error checks (from Kernel 1) with simplified API usage.

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

    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = x.options();
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    // Define tile dimensions for the kernel launch
    const int tile_w = 16;
    const int tile_h = 16;
    dim3 block(tile_w, tile_h);
    dim3 grid((out_width + tile_w - 1) / tile_w, (out_height + tile_h - 1) / tile_h, out_channels);

    // Calculate the shared memory size required per block
    int shared_width = tile_w * stride + (kernel_width - 1) * dilation;
    int shared_height = tile_h * stride + (kernel_height - 1) * dilation;
    size_t shared_mem_size = shared_width * shared_height * sizeof(float);

    tiled_conv2d_kernel<<<grid, block, shared_mem_size>>>(
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
        groups
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled Shared Memory 2D Convolution");
}
