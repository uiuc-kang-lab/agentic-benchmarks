#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for the output
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// This kernel aims to ensure memory coalescing by using 2D loops for loading the shared memory tiles from global memory.
// Threads in the same warp read consecutive memory locations when loading the input patch and the weight kernel.
__global__ void depthwise_conv2d_coalesced_shared_kernel(
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
    // Each block handles a tile corresponding to one (batch, output channel) pair
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;

    // Map output channel to input channel and weight subgroup
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Determine starting indices of the output tile
    int tile_out_x = blockIdx.x * TILE_WIDTH;
    int tile_out_y = blockIdx.y * TILE_HEIGHT;

    // Each thread computes one output coordinate within the tile
    int out_x = tile_out_x + threadIdx.x;
    int out_y = tile_out_y + threadIdx.y;

    // Compute the top-left corner of the input patch for this tile
    int in_start_x = tile_out_x * stride - padding;
    int in_start_y = tile_out_y * stride - padding;

    // Determine shared memory tile size to cover all input pixels needed
    int smem_rows = (TILE_HEIGHT - 1) * stride + kernel_size;
    int smem_cols = (TILE_WIDTH - 1) * stride + kernel_size;

    // Allocate dynamic shared memory: first for input patch, then for the weight kernel
    extern __shared__ float shared_mem[];
    float* s_input  = shared_mem;                        // size: smem_rows * smem_cols
    float* s_weight = shared_mem + smem_rows * smem_cols;  // size: kernel_size * kernel_size

    // Cooperative loading of the weight tile using 2D loops for coalesced access
    for (int r = threadIdx.y; r < kernel_size; r += blockDim.y) {
        for (int c = threadIdx.x; c < kernel_size; c += blockDim.x) {
            int weight_index = r * kernel_size + c;
            s_weight[weight_index] = weight[
                in_ch * (channels_per_group * kernel_size * kernel_size) +
                weight_ch * (kernel_size * kernel_size) +
                weight_index
            ];
        }
    }

    // Cooperative loading of the input patch using 2D loops to ensure coalesced memory accesses
    for (int r = threadIdx.y; r < smem_rows; r += blockDim.y) {
        for (int c = threadIdx.x; c < smem_cols; c += blockDim.x) {
            int global_y = in_start_y + r;
            int global_x = in_start_x + c;
            float value = 0.0f;
            if (global_y >= 0 && global_y < input_h && global_x >= 0 && global_x < input_w) {
                int input_index = b * (in_channels * input_h * input_w) +
                                  in_ch * (input_h * input_w) +
                                  global_y * input_w + global_x;
                value = input[input_index];
            }
            s_input[r * smem_cols + c] = value;
        }
    }

    __syncthreads();

    // Compute the depthwise convolution if the output coordinate is valid
    if (out_y < output_h && out_x < output_w) {
        float sum = 0.0f;
        // Loop over the kernel spatial dimensions
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int s_y = threadIdx.y * stride + ky;
                int s_x = threadIdx.x * stride + kx;
                sum += s_input[s_y * smem_cols + s_x] * s_weight[ky * kernel_size + kx];
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

// Forward function callable from Python via pybind11
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

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Define kernel launch parameters
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    int grid_x = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = batch_size * out_channels;  // each block in z corresponds to one (batch, channel) pair
    dim3 grid(grid_x, grid_y, grid_z);

    // Calculate the shared memory size needed per block
    int smem_rows = (TILE_HEIGHT - 1) * stride + kernel_size;
    int smem_cols = (TILE_WIDTH - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (smem_rows * smem_cols + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_coalesced_shared_kernel<<<grid, block, shared_mem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Coalesced Global Memory Access (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
