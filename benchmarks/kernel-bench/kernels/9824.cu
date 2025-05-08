#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions for output (feel free to adjust for performance)
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// CUDA kernel that uses shared memory for input tiles and weight kernel
__global__ void depthwise_conv2d_shared_kernel(
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
    // Each block handles one tile for a specific (batch, output channel) pair.
    // gridDim.z is batch_size * out_channels.
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;
    
    // Determine corresponding input channel and weight subgroup
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Determine the starting indices for the output tile
    int tile_out_x = blockIdx.x * TILE_WIDTH;
    int tile_out_y = blockIdx.y * TILE_HEIGHT;

    // Each thread computes an output coordinate within the tile
    int out_x = tile_out_x + threadIdx.x;
    int out_y = tile_out_y + threadIdx.y;

    // Compute the top-left corner of the input patch to load into shared memory
    // For an output starting at (tile_out_y, tile_out_x), the corresponding input coordinate is:
    int in_start_y = tile_out_y * stride - padding;
    int in_start_x = tile_out_x * stride - padding;

    // Dimensions of the shared memory tile required for the current block
    // Each output pixel at (ty, tx) accesses input from: (ty*stride + ky, tx*stride + kx)
    int smem_rows = (TILE_HEIGHT - 1) * stride + kernel_size;
    int smem_cols = (TILE_WIDTH - 1) * stride + kernel_size;

    // Allocate dynamic shared memory and partition it
    extern __shared__ float shared_mem[];
    // s_input will cache the input patch for the current tile
    float* s_input = shared_mem;
    // s_weight will store the kernel weight tile (kernel_size x kernel_size)
    float* s_weight = shared_mem + smem_rows * smem_cols;

    // Use a linear index for cooperative loading within the block
    int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;

    // Load the weight tile into shared memory
    int num_weight = kernel_size * kernel_size;
    for (int i = linear_thread; i < num_weight; i += blockDim.x * blockDim.y) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) +
            i
        ];
    }

    // Load the input patch into shared memory
    int total_input = smem_rows * smem_cols;
    for (int i = linear_thread; i < total_input; i += blockDim.x * blockDim.y) {
        int r = i / smem_cols;
        int c = i % smem_cols;
        int global_y = in_start_y + r;
        int global_x = in_start_x + c;
        float val = 0.0f;
        if (global_y >= 0 && global_y < input_h && global_x >= 0 && global_x < input_w) {
            int input_idx = b * (in_channels * input_h * input_w) +
                            in_ch * (input_h * input_w) +
                            global_y * input_w + global_x;
            val = input[input_idx];
        }
        s_input[i] = val;
    }

    __syncthreads();

    // Only compute if the thread's output coordinate is within output bounds
    if (out_y < output_h && out_x < output_w) {
        float sum = 0.0f;
        // Each thread accumulates the convolution sum reading from shared memory
        // The corresponding shared memory indices for the current output pixel:
        // s_y = threadIdx.y * stride + ky, s_x = threadIdx.x * stride + kx
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int s_y = threadIdx.y * stride + ky;
                int s_x = threadIdx.x * stride + kx;
                float in_val = s_input[s_y * smem_cols + s_x];
                float wt = s_weight[ky * kernel_size + kx];
                sum += in_val * wt;
            }
        }

        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[oc];
        }

        int out_idx = b * (out_channels * output_h * output_w) +
                      oc * (output_h * output_w) +
                      out_y * output_w +
                      out_x;
        output[out_idx] = sum;
    }
}


// Forward function called from Python via pybind11
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

    // Define kernel launch dimensions
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    int grid_x = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = batch_size * out_channels; // Each block in z corresponds to one (batch, channel) pair
    dim3 grid(grid_x, grid_y, grid_z);

    // Calculate shared memory size required per block
    int smem_rows = (TILE_HEIGHT - 1) * stride + kernel_size;
    int smem_cols = (TILE_WIDTH - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (smem_rows * smem_cols + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_shared_kernel<<<grid, block, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Shared Memory (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
