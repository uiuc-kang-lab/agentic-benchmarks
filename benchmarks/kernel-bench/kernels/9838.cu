#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Tuning parameter: total number of threads per block (must be a perfect square or near-perfect square for tiling)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// CUDA kernel using shared memory with tunable block size
__global__ void depthwise_conv2d_shared_tuned_kernel(
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
    // Use blockDim.x as the tile dimension; assume blockDim.x == blockDim.y
    const int tile_dim = blockDim.x; 

    // Determine which (batch, channel) this block is responsible for
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Compute output tile start indices
    int tile_out_x = blockIdx.x * tile_dim;
    int tile_out_y = blockIdx.y * tile_dim;

    // Each thread computes one output element in the tile
    int out_x = tile_out_x + threadIdx.x;
    int out_y = tile_out_y + threadIdx.y;

    // Compute corresponding input top-left coordinate
    int in_start_x = tile_out_x * stride - padding;
    int in_start_y = tile_out_y * stride - padding;

    // Shared memory tile dimensions for input patch
    int smem_cols = (tile_dim - 1) * stride + kernel_size;
    int smem_rows = (tile_dim - 1) * stride + kernel_size;

    // Allocate dynamic shared memory: first for input patch, then for kernel weights
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem; // size = smem_rows * smem_cols
    float* s_weight = shared_mem + smem_rows * smem_cols; // size = kernel_size * kernel_size

    // Cooperative loading using a linear thread index
    int linear_thread = threadIdx.y * tile_dim + threadIdx.x;
    int total_threads = tile_dim * tile_dim;

    // Load kernel weights into shared memory
    int num_weight = kernel_size * kernel_size;
    for (int i = linear_thread; i < num_weight; i += total_threads) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) +
            i
        ];
    }

    // Load input patch into shared memory
    int total_input = smem_rows * smem_cols;
    for (int i = linear_thread; i < total_input; i += total_threads) {
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

    if (out_y < output_h && out_x < output_w) {
        float sum = 0.0f;
        // Compute convolution: each thread computes one output pixel using shared memory
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
        int out_idx = b * (out_channels * output_h * output_w) +
                      oc * (output_h * output_w) +
                      out_y * output_w + out_x;
        output[out_idx] = sum;
    }
}

// Forward function callable from Python
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

    // Determine block (tile) dimensions based on BLOCK_SIZE
    int tile_dim = static_cast<int>(std::floor(std::sqrt((float)BLOCK_SIZE)));
    dim3 block(tile_dim, tile_dim);
    
    int grid_x = (output_w + tile_dim - 1) / tile_dim;
    int grid_y = (output_h + tile_dim - 1) / tile_dim;
    int grid_z = batch_size * out_channels;
    dim3 grid(grid_x, grid_y, grid_z);

    // Calculate required shared memory size per block
    int smem_cols = (tile_dim - 1) * stride + kernel_size;
    int smem_rows = (tile_dim - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (smem_rows * smem_cols + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    
    depthwise_conv2d_shared_tuned_kernel<<<grid, block, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Tunable Block Size (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
