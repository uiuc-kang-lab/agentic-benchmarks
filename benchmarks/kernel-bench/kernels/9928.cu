#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory tiling to improve memory coalescing for global memory accesses.
// Each block computes one output tile for a specific (batch, output channel) pair.
// The input tile for the block is loaded into shared memory in a coalesced manner, and the small convolution filter is loaded into a static shared array.

// Assumptions:
// - The data is in NCHW layout (batch, channel, height, width).
// - blockDim.x and blockDim.y represent the output tile dimensions (TILE_W x TILE_H).
// - Kernel filters are small (kernel_size*kernel_size <= 64).

// Kernel: depthwise_shared_coalesced_kernel
__global__ void depthwise_shared_coalesced_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    // Each block is assigned to a unique (batch, output channel) pair
    int blockId = blockIdx.z;  // blockIdx.z in [0, batch_size * out_channels)
    int b = blockId / out_channels;
    int oc = blockId % out_channels;
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Load the convolution kernel (filter) into static shared memory
    __shared__ float sKernel[64];  // supports kernel_size*kernel_size up to 64 elements
    int kernel_elems = kernel_size * kernel_size;
    if (threadIdx.y == 0 && threadIdx.x < kernel_elems) {
        int weight_index = in_ch * (channels_per_group * kernel_elems) +
                           weight_ch * kernel_elems + threadIdx.x;
        sKernel[threadIdx.x] = weight[weight_index];
    }
    __syncthreads();

    // Define output tile dimensions based on block dimensions
    int tile_out_w = blockDim.x;  // number of output columns computed by the block
    int tile_out_h = blockDim.y;  // number of output rows computed by the block

    // Compute the required shared memory tile dimensions for input
    // For each output pixel, we need a kernel_size window; with stride, the height of the
    // input tile is: tile_out_h * stride + (kernel_size - stride).
    int smem_h = tile_out_h * stride + (kernel_size - stride);
    int smem_w = tile_out_w * stride + (kernel_size - stride);

    // Determine the top-left corner of the output tile in output space
    int out_x0 = blockIdx.x * tile_out_w;
    int out_y0 = blockIdx.y * tile_out_h;

    // Corresponding top-left coordinate in the input (taking stride and padding into account)
    int in_x0 = out_x0 * stride - padding;
    int in_y0 = out_y0 * stride - padding;

    // Allocate dynamic shared memory for the input tile
    extern __shared__ float sInput[];  // size: smem_h * smem_w floats
    int total_smem = smem_h * smem_w;

    // Each thread loads elements into shared memory in a coalesced manner
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;
    for (int idx = thread_id; idx < total_smem; idx += block_threads) {
        int i = idx / smem_w;
        int j = idx % smem_w;
        int in_y = in_y0 + i;
        int in_x = in_x0 + j;
        float val = 0.0f;
        if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
            int input_index = b * (in_channels * input_h * input_w) +
                              in_ch * (input_h * input_w) +
                              in_y * input_w + in_x;
            val = input[input_index];
        }
        sInput[idx] = val;
    }
    __syncthreads();

    // Each thread computes its corresponding output pixel in the tile
    int out_x = out_x0 + threadIdx.x;
    int out_y = out_y0 + threadIdx.y;
    if (out_x < output_w && out_y < output_h) {
        float sum = 0.0f;
        // The position within shared memory from where the convolution window starts for this output
        int local_y = threadIdx.y * stride;
        int local_x = threadIdx.x * stride;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int s_y = local_y + kh;
                int s_x = local_x + kw;
                float in_val = sInput[s_y * smem_w + s_x];
                float w_val = sKernel[kh * kernel_size + kw];
                sum += in_val * w_val;
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int out_index = b * (out_channels * output_h * output_w) +
                        oc * (output_h * output_w) +
                        out_y * output_w + out_x;
        output[out_index] = sum;
    }
}


// Forward function called from PyTorch
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Define tile dimensions for output (block dimensions) - these can be tuned
    const int TILE_W = 16;
    const int TILE_H = 16;
    dim3 blockDim(TILE_W, TILE_H, 1);

    // Grid dimensions:
    // - gridDim.x covers the output width tiles
    // - gridDim.y covers the output height tiles
    // - gridDim.z covers all (batch, output channel) combinations
    dim3 gridDim((output_w + TILE_W - 1) / TILE_W,
                 (output_h + TILE_H - 1) / TILE_H,
                 batch_size * out_channels);

    // Compute shared memory size for the input tile
    int smem_h = TILE_H * stride + (kernel_size - stride);
    int smem_w = TILE_W * stride + (kernel_size - stride);
    size_t shared_mem_size = smem_h * smem_w * sizeof(float);

    depthwise_shared_coalesced_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        output_h,
        output_w,
        kernel_size,
        stride,
        padding,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution with shared memory tiling and coalesced accesses",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
