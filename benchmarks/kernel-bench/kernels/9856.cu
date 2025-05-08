#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for the output tile. We choose 16x16 for this design.
#define TILE_OUT_WIDTH 16
#define TILE_OUT_HEIGHT 16

// This kernel partitions the convolution computation for one (batch, channel) pair over a tile of output pixels.
// Each block processes one tile of the output feature map for a given (batch, oc) pair. The computation of each output
// element is split across the kernel window, with partial contributions accumulated into a shared memory
// accumulator using atomic operations. These atomicAdd calls are performed in shared memory (which is fast)
// and are used only to handle race conditions when multiple thread iterations contribute to the same output
// accumulator. Finally, the tile results (with optional bias addition) are written out to global memory with a
// single non-atomic store per output element.

__global__ void depthwise_conv2d_atomic_shared_kernel(
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
    // Determine the batch and output channel from blockIdx.z
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;

    // Map output channel back to input channel and the corresponding weight channel
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Determine the starting coordinates of the tile processed by this block
    int tile_out_x = blockIdx.x * TILE_OUT_WIDTH;
    int tile_out_y = blockIdx.y * TILE_OUT_HEIGHT;

    // Total number of pixels in the tile
    const int tile_size = TILE_OUT_WIDTH * TILE_OUT_HEIGHT;

    // Allocate shared memory accumulator for the tile (dynamically allocated).
    // Each element corresponds to one output pixel accumulator for the tile.
    extern __shared__ float shared_accum[]; // size: tile_size * sizeof(float)

    // Initialize the shared memory accumulator to 0
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        shared_accum[i] = 0.0f;
    }
    __syncthreads();

    // Total number of kernel elements
    int total_kernel = kernel_size * kernel_size;

    // Loop over each element in the convolution kernel
    for (int op = 0; op < total_kernel; op++) {
        int ky = op / kernel_size;
        int kx = op % kernel_size;

        // Each thread processes a subset of the tile's output pixels
        for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
            int local_y = idx / TILE_OUT_WIDTH;
            int local_x = idx % TILE_OUT_WIDTH;
            int out_x = tile_out_x + local_x;
            int out_y = tile_out_y + local_y;

            // Compute the corresponding input coordinates for this kernel element
            int in_x = out_x * stride + kx - padding;
            int in_y = out_y * stride + ky - padding;

            float val = 0.0f;
            if (out_x < output_w && out_y < output_h && in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
                int input_idx = b * (in_channels * input_h * input_w) +
                                in_ch * (input_h * input_w) +
                                in_y * input_w + in_x;
                val = input[input_idx];
            }

            // Get the corresponding weight value
            int weight_idx = in_ch * (channels_per_group * total_kernel) +
                             weight_ch * total_kernel +
                             op;
            float prod = val * weight[weight_idx];

            // Use atomicAdd in shared memory to accumulate the partial result for this output pixel
            atomicAdd(&shared_accum[idx], prod);
        }
    }
    __syncthreads();

    // Write the accumulated results from shared memory to global memory
    for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
        int local_y = idx / TILE_OUT_WIDTH;
        int local_x = idx % TILE_OUT_WIDTH;
        int out_x = tile_out_x + local_x;
        int out_y = tile_out_y + local_y;
        if (out_x < output_w && out_y < output_h) {
            float sum = shared_accum[idx];
            if (bias != nullptr) {
                sum += bias[oc];
            }
            int out_idx = b * (out_channels * output_h * output_w) +
                          oc * (output_h * output_w) +
                          out_y * output_w + out_x;
            output[out_idx] = sum;
        }
    }
}

// Forward function callable from Python via pybind11
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Input and weight must be CUDA tensors");
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

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Grid dimensions: each block processes a tile of output pixels for one (batch, channel) pair
    int grid_x = (output_w + TILE_OUT_WIDTH - 1) / TILE_OUT_WIDTH;
    int grid_y = (output_h + TILE_OUT_HEIGHT - 1) / TILE_OUT_HEIGHT;
    int grid_z = batch_size * out_channels;
    dim3 grid(grid_x, grid_y, grid_z);

    // Use 256 threads per block
    int blockSize = 256;
    dim3 block(blockSize);

    // Shared memory size (in bytes): one float per tile pixel
    size_t shared_mem_size = TILE_OUT_WIDTH * TILE_OUT_HEIGHT * sizeof(float);

    depthwise_conv2d_atomic_shared_kernel<<<grid, block, shared_mem_size>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Shared Memory Atomic Accumulation (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
