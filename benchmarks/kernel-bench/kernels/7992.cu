#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Assume maximum weight and bias sizes for constant memory usage.
// Adjust these sizes as needed for your application.
__constant__ float d_weight_const[12288];  // Constant memory for weights
__constant__ float d_bias_const[1024];     // Constant memory for bias

// This kernel uses shared memory to cache input tiles and minimizes __syncthreads() usage.
// Each block loads a tile of the input corresponding to the block's output region.
// Threads synchronize only once after loading the shared memory tile and once after finishing the per-channel computation,
// ensuring correctness without excessive synchronizations.

extern __shared__ float s_tile[];  // Dynamic shared memory for input tile

__global__ void conv2d_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width) {

    // Determine batch index and output channel from merged z-dimension
    int bc = blockIdx.z;  // combined index: bc = b * out_channels + oc
    int b  = bc / out_channels;
    int oc = bc % out_channels;

    // Compute block output start indices in spatial dimensions
    int out_start_x = blockIdx.x * blockDim.x;
    int out_start_y = blockIdx.y * blockDim.y;

    // Each thread's output coordinate within the full output
    int out_x = out_start_x + threadIdx.x;
    int out_y = out_start_y + threadIdx.y;

    // Compute the corresponding input tile origin for this block
    int in_tile_origin_x = out_start_x * stride - padding;
    int in_tile_origin_y = out_start_y * stride - padding;

    // Determine the dimensions of the shared memory tile
    int tile_width  = blockDim.x * stride + (kernel_size - 1) * dilation;
    int tile_height = blockDim.y * stride + (kernel_size - 1) * dilation;
    int tile_size = tile_width * tile_height;

    float sum = 0.0f;

    // Loop over all input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load the current input channel's tile into shared memory
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = tid; i < tile_size; i += blockDim.x * blockDim.y) {
            int tx = i % tile_width;
            int ty = i / tile_width;
            int in_x = in_tile_origin_x + tx;
            int in_y = in_tile_origin_y + ty;
            float val = 0.0f;
            if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                int input_idx = ((b * in_channels + ic) * in_height + in_y) * in_width + in_x;
                val = input[input_idx];
            }
            s_tile[i] = val;
        }
        // Synchronize to ensure the entire tile is loaded before computation
        __syncthreads();

        // Only compute if the output coordinates are within range
        if (out_x < out_width && out_y < out_height) {
            // Determine the local starting coordinate in shared memory for this output pixel
            // The shared memory tile corresponds to input pixels starting at (in_tile_origin_x, in_tile_origin_y).
            // For the output pixel at (out_x, out_y), the corresponding input location is out_x*stride, out_y*stride.
            // Thus, its position in shared memory is:
            int local_x = (out_x * stride) - in_tile_origin_x;  // equivalent to threadIdx.x * stride + padding
            int local_y = (out_y * stride) - in_tile_origin_y;  // equivalent to threadIdx.y * stride + padding

            // Loop over the kernel window and accumulate the products
            for (int kh = 0; kh < kernel_size; ++kh) {
                int s_y = local_y + kh * dilation;
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int s_x = local_x + kw * dilation;
                    int tile_idx = s_y * tile_width + s_x;
                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    float w = __ldg(&d_weight_const[weight_idx]);
                    sum += s_tile[tile_idx] * w;
                }
            }
        }
        // Synchronize before reusing shared memory for the next input channel
        __syncthreads();
    }

    // Write the result (with bias added) if within output bounds
    if (out_x < out_width && out_y < out_height) {
        float bias_val = __ldg(&d_bias_const[oc]);
        sum += bias_val;
        int output_idx = ((b * out_channels + oc) * out_height + out_y) * out_width + out_x;
        output[output_idx] = sum;
    }
}

// Host function to launch the shared memory optimized convolution kernel

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

    TORCH_CHECK(groups == 1, "shared_mem_conv2d kernel supports groups==1 only");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);  // assumes square kernel

    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    // Copy weights and bias into constant memory
    cudaMemcpyToSymbol(d_weight_const, weight.data_ptr<float>(), weight.numel() * sizeof(float));
    if (bias.has_value()) {
        cudaMemcpyToSymbol(d_bias_const, bias.value().data_ptr<float>(), bias.value().numel() * sizeof(float));
    } else {
        std::vector<float> zeros(out_channels, 0.0f);
        cudaMemcpyToSymbol(d_bias_const, zeros.data(), out_channels * sizeof(float));
    }

    // Define block and grid dimensions for spatial tiling and merging batch/out_channels into z-dimension
    dim3 block(16, 16);
    dim3 grid(
        (out_width  + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch_size * out_channels
    );

    // Compute dynamic shared memory size needed per block
    int tile_width  = block.x * stride + (kernel_size - 1) * dilation;
    int tile_height = block.y * stride + (kernel_size - 1) * dilation;
    int shared_mem_size = tile_width * tile_height * sizeof(float);

    conv2d_shared_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        out_height,
        out_width
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "2D convolution with shared memory and minimal __syncthreads() usage");
}
