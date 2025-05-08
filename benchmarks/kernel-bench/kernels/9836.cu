#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions (can be tuned further)
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Optimized CUDA kernel that fuses shared memory tiling with loop unrolling
__global__ void depthwise_conv2d_tiled_shared_kernel(
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
    // Each block is responsible for a tile of output pixels for a specific (batch, output channel) pair
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;

    // Map output channel to input channel and corresponding weight subgroup
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Compute the output coordinate for this thread
    int out_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int out_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;

    // Determine shared memory tile dimensions for the input patch
    int smem_width  = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;

    // Compute the top-left corner in the input corresponding to this output tile
    // Note: For an output tile starting at (tile_out_x, tile_out_y), where tile_out_x = blockIdx.x * TILE_WIDTH
    // the corresponding input coordinate is: (tile_out_x * stride - padding, tile_out_y * stride - padding)
    int in_start_x = blockIdx.x * TILE_WIDTH * stride - padding;
    int in_start_y = blockIdx.y * TILE_HEIGHT * stride - padding;

    // Allocate dynamic shared memory: first for input patch, then kernel weights
    extern __shared__ float shared_mem[];
    float* s_input  = shared_mem;  // size: smem_width * smem_height
    float* s_weight = shared_mem + smem_width * smem_height;  // size: kernel_size * kernel_size

    // Cooperative loading using a linear index within the block
    int tId = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;

    // Load kernel weights into shared memory (only once per block)
    int num_weight = kernel_size * kernel_size;
    for (int i = tId; i < num_weight; i += blockSize) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) +
            i
        ];
    }

    // Load the input patch for this output tile into shared memory
    int total_input = smem_width * smem_height;
    for (int i = tId; i < total_input; i += blockSize) {
        int r = i / smem_width;
        int c = i % smem_width;
        int global_y = in_start_y + r;
        int global_x = in_start_x + c;
        float val = 0.0f;
        if (global_y >= 0 && global_y < input_h && global_x >= 0 && global_x < input_w) {
            int input_idx = b * (in_channels * input_h * input_w) +
                            in_ch * (input_h * input_w) +
                            global_y * input_w +
                            global_x;
            val = input[input_idx];
        }
        s_input[i] = val;
    }

    __syncthreads();

    // Only compute if the thread's output coordinate is within bounds
    if (out_x < output_w && out_y < output_h) {
        float sum = 0.0f;
        // Use loop unrolling for small kernel sizes to boost performance
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                // Compute the corresponding shared memory indices
                int s_y = threadIdx.y * stride + ky;
                int s_x = threadIdx.x * stride + kx;
                float inp_val = s_input[s_y * smem_width + s_x];
                float wt_val = s_weight[ky * kernel_size + kx];
                sum += inp_val * wt_val;
            }
        }

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

    int batch_size   = input.size(0);
    int in_channels  = input.size(1);
    int input_h      = input.size(2);
    int input_w      = input.size(3);
    int kernel_size  = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Define grid dimensions: Each block processes a TILE_WIDTH x TILE_HEIGHT patch for one (batch, channel)
    int grid_x = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = batch_size * out_channels;
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(TILE_WIDTH, TILE_HEIGHT);

    // Calculate shared memory size: input patch + kernel weights
    int smem_width  = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (smem_width * smem_height + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_tiled_shared_kernel<<<grid, block, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Optimized Depthwise 2D Convolution with Tiling and Shared Memory (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
