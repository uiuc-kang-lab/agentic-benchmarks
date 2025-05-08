#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory to cache input tiles for each input channel
// to reduce global memory latency.
// Each block computes a tile of the output for a fixed batch and output channel.

__global__ void shared_conv2d_kernel(
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

    // Tile dimensions are defined by block dimensions (each thread corresponds to one output pixel)
    const int tile_w = blockDim.x;  // number of threads in x-direction
    const int tile_h = blockDim.y;  // number of threads in y-direction

    // Determine the batch and output channel this block is working on
    int global_block = blockIdx.z; // ranges from 0 to batch_size*out_channels - 1
    int b = global_block / out_channels;
    int oc = global_block % out_channels;

    // Compute the starting output coordinates for this block
    int tile_out_x = blockIdx.x * tile_w;
    int tile_out_y = blockIdx.y * tile_h;

    // Global output coordinates for this thread
    int out_x = tile_out_x + threadIdx.x;
    int out_y = tile_out_y + threadIdx.y;

    // Compute dimensions of the shared memory tile for one input channel
    // Each output pixel uses input starting at (out*y * stride) plus kernel extent.
    int smem_w = tile_w * stride + (kernel_w - 1) * dilation_w;
    int smem_h = tile_h * stride + (kernel_h - 1) * dilation_h;

    // The starting global input coordinates for the shared memory tile
    int in_tile_start_x = tile_out_x * stride - pad_w;
    int in_tile_start_y = tile_out_y * stride - pad_h;

    // Declare dynamic shared memory
    extern __shared__ float smem[]; // size smem_h * smem_w floats

    float sum = 0.0f;
    if (bias) {
        sum = bias[oc];
    }

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Pointer to the current input channel for batch b
        const float* x_ptr = x + (b * in_channels + ic) * input_height * input_width;

        // Load the required patch of the input into shared memory.
        // Each thread loads multiple elements in a strided pattern
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int block_size = blockDim.x * blockDim.y;
        
        // Ensure all threads participate in loading shared memory
        for (int y = 0; y < smem_h; y++) {
            for (int x = tid; x < smem_w; x += block_size) {
                int in_x = in_tile_start_x + x;
                int in_y = in_tile_start_y + y;
                float val = 0.0f;
                
                // Bounds checking for input array
                if (in_x >= 0 && in_x < input_width && 
                    in_y >= 0 && in_y < input_height) {
                    val = x_ptr[in_y * input_width + in_x];
                }
                
                // Write to shared memory
                smem[y * smem_w + x] = val;
            }
        }
        __syncthreads();

        // Only compute if this thread corresponds to a valid output element
        if (out_x < width_out && out_y < height_out) {
            // For each element in the kernel
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // Compute the shared memory indices corresponding to the input element used for this kernel element
                    // Global input coordinate for this kernel application:
                    // global_x = out_x * stride + kw * dilation_w
                    // global_y = out_y * stride + kh * dilation_h
                    // Shared memory coordinate is offset by the starting input coordinate of the tile:
                    int local_x = (threadIdx.x * stride + kw * dilation_w) + pad_w; // since in_tile_start_x = tile_out_x*stride - pad_w
                    int local_y = (threadIdx.y * stride + kh * dilation_h) + pad_h;
                    int smem_index = local_y * smem_w + local_x;

                    // Weight layout: [out_channels, in_channels, kernel_h, kernel_w]
                    int weight_index = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;

                    sum += smem[smem_index] * weight[weight_index];
                }
            }
        }
        __syncthreads(); // Re-sync before loading next input channel
    }

    // Write the output if within bounds
    if (out_x < width_out && out_y < height_out) {
        int out_index = ((b * out_channels + oc) * height_out + out_y) * width_out + out_x;
        output[out_index] = sum;
    }
}


// Forward function called from PyTorch

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

    // Compute output dimensions
    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    if (batch_size * out_channels * height_out * width_out == 0) return output;

    // Define tile (block) size; these can be tuned for optimum performance
    int tile_h = 16;
    int tile_w = 16;
    dim3 block(tile_w, tile_h);

    // Grid dimensions: tile the output spatial dimensions and account for batch and out_channels
    int grid_x = (width_out + tile_w - 1) / tile_w;
    int grid_y = (height_out + tile_h - 1) / tile_h;
    int grid_z = batch_size * out_channels; // one block per (batch, out_channel) pair
    dim3 grid(grid_x, grid_y, grid_z);

    // Compute shared memory size per block
    int smem_w = tile_w * stride + (kernel_w - 1) * dilation_w;
    int smem_h = tile_h * stride + (kernel_h - 1) * dilation_h;
    size_t shared_mem_size = smem_w * smem_h * sizeof(float);

    // Launch the kernel
    shared_conv2d_kernel<<<grid, block, shared_mem_size>>>(
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
    m.def("forward", &forward, "Conv2D forward with shared memory (CUDA)");
}
