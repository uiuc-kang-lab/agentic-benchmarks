/*
This CUDA kernel combines shared memory tiling (from Kernel 1) with vectorized output computation (from Kernel 2).
Each block loads a large input tile (with halo) into shared memory and then each thread computes a vector of output elements (vector length V).
The shared tile reduces global memory accesses while vectorized computation enhances memory coalescing and throughput.

Assumptions:
- TILE_SIZE defines the number of output rows per block (16 here).
- V defines the number of output elements computed per thread in the x direction (set to 4).
- Grid dimensions are chosen so that each block covers a tile of (TILE_SIZE*V) outputs in x and TILE_SIZE outputs in y.

This kernel works for general stride/padding/dilation parameters.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define V 4  // vectorization factor (outputs per thread in x)

// The kernel: each thread computes V contiguous outputs in x
__global__ void depthwise_conv2d_vectorized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    // We assume gridDim.z is organized as (batch * out_channels), similar to Kernel 1/Kernel 2
    int block_channel = blockIdx.z;
    int b = block_channel / out_channels;
    int c = block_channel % out_channels;
    int g = c / channels_per_group;
    int m = c % channels_per_group;

    // Each block covers a tile of outputs:
    //   out_tile_width = TILE_SIZE * V (in x direction)
    //   out_tile_height = TILE_SIZE (in y direction)
    const int out_tile_width = TILE_SIZE * V;
    const int out_tile_height = TILE_SIZE;

    // Compute the top-left corner of the output tile in global coordinates
    int global_tile_out_x = blockIdx.x * out_tile_width;
    int global_tile_out_y = blockIdx.y * out_tile_height;

    // The corresponding top-left corner in the input:
    // Note: For depthwise conv, each output pixel uses an input patch starting at (out * stride - padding)
    int in_x_origin = global_tile_out_x * stride_w - padding_w;
    int in_y_origin = global_tile_out_y * stride_h - padding_h;

    // Compute the required shared memory tile dimensions
    // For x: the last output in the block is at global x = global_tile_out_x + out_tile_width - 1
    // Its receptive field goes till: ( (global_tile_out_x + out_tile_width - 1)*stride + (kernel_w - 1)*dilation )
    // We add 1 to get the proper width.
    int tile_width = (out_tile_width - 1) * stride_w + (kernel_w - 1) * dilation_w + 1;
    int tile_height = (out_tile_height - 1) * stride_h + (kernel_h - 1) * dilation_h + 1;

    // Allocate shared memory tile
    extern __shared__ float shared_input[];  // size = tile_width * tile_height
    int tile_size = tile_width * tile_height;

    // Load the shared memory tile from global memory in a coalesced manner.
    // Use a 1D loop over the tile's linear indices.
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < tile_size; idx += blockDim.x * blockDim.y) {
        int tile_i = idx / tile_width;
        int tile_j = idx % tile_width;
        int in_y = in_y_origin + tile_i;
        int in_x = in_x_origin + tile_j;
        float val = 0.0f;
        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
            // Compute input index. For depthwise conv, each channel picks its own filter.
            // Here, input channel is chosen as: for group g, we select channel = g. (Assuming depthwise scenario)
            int input_idx = b * in_channels * in_h * in_w + (g) * in_h * in_w + in_y * in_w + in_x;
            val = input[input_idx];
        }
        shared_input[idx] = val;
    }

    __syncthreads();

    // Each thread computes V outputs for a specific out y coordinate.
    // Determine the output y coordinate for this thread
    int ty = threadIdx.y;
    int global_out_y = global_tile_out_y + ty;

    // For x, each thread computes V outputs starting at column: global_tile_out_x + threadIdx.x * V
    int tx_base = threadIdx.x * V;

    // Accumulators for V outputs
    float sum[V];
    #pragma unroll
    for (int i = 0; i < V; i++) {
        sum[i] = 0.0f;
    }

    // Pre-compute the base shared memory y-offset to reduce register usage
    const int base_sh_y = ty * stride_h;
    
    // Loop over the kernel's height and width
    #pragma unroll
    for (int kh = 0; kh < kernel_h; kh++) {
        const int sh_y = base_sh_y + kh * dilation_h;
        const int weight_row_offset = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w;
        
        #pragma unroll
        for (int kw = 0; kw < kernel_w; kw++) {
            // Get the weight value for this kernel element - compute index more efficiently
            float w_val = weight[weight_row_offset + kw];
            
            // Pre-compute the dilation offset for the current kernel position
            const int x_dilation_offset = kw * dilation_w;
            
            // Each thread processes V outputs horizontally
            #pragma unroll
            for (int v = 0; v < V; v++) {
                // Compute shared memory index more efficiently
                const int sh_x = (tx_base + v) * stride_w + x_dilation_offset;
                const int sh_index = sh_y * tile_width + sh_x;
                sum[v] += shared_input[sh_index] * w_val;
            }
        }
    }

    // Write the computed outputs to global memory, with boundary checks
    if (global_out_y < out_h) {
        #pragma unroll
        for (int v = 0; v < V; v++) {
            int global_out_x = global_tile_out_x + tx_base + v;
            if (global_out_x < out_w) {
                int out_index = ((b * out_channels + c) * out_h + global_out_y) * out_w + global_out_x;
                float res = sum[v];
                if (bias != nullptr) {
                    res += bias[c];
                }
                output[out_index] = res;
            }
        }
    }
}


// Host function: forward pass
// It computes grid and block dimensions and allocates appropriate shared memory.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    // Compute output spatial dimensions
    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    // Define block dimensions
    dim3 threads(TILE_SIZE, TILE_SIZE);

    // Each block produces an output tile of dimensions: 
    //   width: TILE_SIZE * V (in x)
    //   height: TILE_SIZE (in y)
    int out_tile_width = TILE_SIZE * V;
    int grid_x = (out_w + out_tile_width - 1) / out_tile_width;
    int grid_y = (out_h + TILE_SIZE - 1) / TILE_SIZE;
    int grid_z = batch_size * out_channels;  // one block per (batch, channel)
    dim3 blocks(grid_x, grid_y, grid_z);

    // Compute shared memory tile size (in bytes).
    // The shared tile covers the input region for the entire output tile.
    int tile_width = (out_tile_width - 1) * stride_w + (kernel_w - 1) * dilation_w + 1;
    int tile_height = (TILE_SIZE - 1) * stride_h + (kernel_h - 1) * dilation_h + 1;
    size_t shared_mem_size = tile_width * tile_height * sizeof(float);

    depthwise_conv2d_vectorized_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise Conv2D forward (CUDA) - Shared Memory with Vectorized Output");
}
