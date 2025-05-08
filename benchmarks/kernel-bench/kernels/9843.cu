#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions for output. These can be tuned for performance.
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Optimized CUDA kernel with manual loop unrolling for kernel_size==3
// and #pragma unroll for other kernel sizes.
__global__ void depthwise_conv2d_manual_unroll_kernel(
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
    // Determine which (batch, out_channel) this block is processing.
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;
    
    // Map output channel to input channel and corresponding weight subgroup
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Calculate the starting indices for the output tile
    int tile_out_x = blockIdx.x * TILE_WIDTH;
    int tile_out_y = blockIdx.y * TILE_HEIGHT;

    // Each thread computes one output coordinate within the tile
    int out_x = tile_out_x + threadIdx.x;
    int out_y = tile_out_y + threadIdx.y;

    // Compute the top-left corner of the corresponding input patch
    int in_start_x = tile_out_x * stride - padding;
    int in_start_y = tile_out_y * stride - padding;

    // Dimensions of the shared memory tile needed to cover the output tile
    int smem_cols = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_rows = (TILE_HEIGHT - 1) * stride + kernel_size;

    // Allocate dynamic shared memory: first for input patch then for kernel weights
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;                    // size = smem_rows * smem_cols
    float* s_weight = shared_mem + smem_rows * smem_cols; // size = kernel_size * kernel_size

    // Use a linear index for cooperative loading
    int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;

    // Load the kernel weights into shared memory
    int num_weight = kernel_size * kernel_size;
    for (int i = linear_thread; i < num_weight; i += block_threads) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) +
            i
        ];
    }

    // Load the input patch into shared memory
    int total_input = smem_rows * smem_cols;
    for (int i = linear_thread; i < total_input; i += block_threads) {
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

    // Only compute if the thread's output coordinate is valid
    if (out_y < output_h && out_x < output_w) {
        float sum = 0.0f;
        
        // Use manual unrolling for the common 3x3 kernel case to reduce loop overhead
        if (kernel_size == 3) {
            // Compute starting index in shared memory for this thread
            int s_row = threadIdx.y * stride;
            int s_col = threadIdx.x * stride;
            
            sum += s_input[s_row * smem_cols + s_col] * s_weight[0];
            sum += s_input[s_row * smem_cols + s_col + 1] * s_weight[1];
            sum += s_input[s_row * smem_cols + s_col + 2] * s_weight[2];
            
            sum += s_input[(s_row + 1) * smem_cols + s_col] * s_weight[3];
            sum += s_input[(s_row + 1) * smem_cols + s_col + 1] * s_weight[4];
            sum += s_input[(s_row + 1) * smem_cols + s_col + 2] * s_weight[5];
            
            sum += s_input[(s_row + 2) * smem_cols + s_col] * s_weight[6];
            sum += s_input[(s_row + 2) * smem_cols + s_col + 1] * s_weight[7];
            sum += s_input[(s_row + 2) * smem_cols + s_col + 2] * s_weight[8];
        } else {
            // For other kernel sizes, use #pragma unroll to guide compile-time unrolling
            #pragma unroll
            for (int ky = 0; ky < kernel_size; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int s_y = threadIdx.y * stride + ky;
                    int s_x = threadIdx.x * stride + kx;
                    sum += s_input[s_y * smem_cols + s_x] * s_weight[ky * kernel_size + kx];
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[oc];
        }

        int out_idx = b * (out_channels * output_h * output_w) +
                      oc * (output_h * output_w) +
                      out_y * output_w + out_x;
        output[out_idx] = sum;
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

    // Define grid and block dimensions
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    int grid_x = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = batch_size * out_channels; // Each block in z corresponds to one (batch, channel) pair
    dim3 grid(grid_x, grid_y, grid_z);

    // Calculate shared memory size required per block
    int smem_cols = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_rows = (TILE_HEIGHT - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (smem_rows * smem_cols + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_manual_unroll_kernel<<<grid, block, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Manual Unrolling (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
