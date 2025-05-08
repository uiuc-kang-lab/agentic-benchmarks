#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for the output
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Optimized kernel combining shared memory tiling and efficient grid mapping
__global__ void depthwise_conv2d_opt_kernel(
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
    int channels_per_group) {

    // Each block is responsible for one (batch, out_channel) pair
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;
    
    // Determine corresponding input channel and weight subgroup
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Calculate the top-left corner of the output tile for this block
    int tile_out_x = blockIdx.x * TILE_WIDTH;
    int tile_out_y = blockIdx.y * TILE_HEIGHT;

    // Global output coordinate computed by each thread
    int out_x = tile_out_x + threadIdx.x;
    int out_y = tile_out_y + threadIdx.y;

    // Determine the size of the required shared memory tile for the input patch
    // (TILE_WIDTH-1)*stride + kernel_size covers the span of input pixels needed
    int smem_width  = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;

    // Starting coordinates in the input corresponding to the top-left of the shared memory tile
    int in_start_x = tile_out_x * stride - padding;
    int in_start_y = tile_out_y * stride - padding;

    // Allocate dynamic shared memory: first for input patch, then for kernel weights
    extern __shared__ float shared_mem[];
    float* s_input  = shared_mem;                           // size = smem_width * smem_height
    float* s_weight = shared_mem + smem_width * smem_height;  // size = kernel_size * kernel_size

    // Use a linear index within the block for cooperative loading
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    // Load the kernel weights into shared memory (each block loads its corresponding filter for this channel)
    int num_weight = kernel_size * kernel_size;
    for (int i = tid; i < num_weight; i += total_threads) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) + i
        ];
    }

    // Load the required input patch into shared memory
    int total_input = smem_width * smem_height;
    for (int i = tid; i < total_input; i += total_threads) {
        int r = i / smem_width;
        int c = i % smem_width;
        int in_y = in_start_y + r;
        int in_x = in_start_x + c;
        float val = 0.0f;
        if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
            int input_idx = b * (in_channels * input_h * input_w) +
                            in_ch * (input_h * input_w) +
                            in_y * input_w + in_x;
            val = input[input_idx];
        }
        s_input[i] = val;
    }

    __syncthreads();

    // Only compute if the thread's output coordinate is valid
    if (out_x < output_w && out_y < output_h) {
        float sum = 0.0f;
        // Perform the convolution using the values stored in shared memory
        // Each thread computes one output pixel
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // The corresponding shared memory index for the input pixel
                int s_y = threadIdx.y * stride + ky;
                int s_x = threadIdx.x * stride + kx;
                sum += s_input[s_y * smem_width + s_x] * s_weight[ky * kernel_size + kx];
            }
        }
        // Add bias if provided
        if (bias)
            sum += bias[oc];

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
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Input and weight must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor if provided");
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

    if(bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Setup grid: each block handles a TILE_WIDTH x TILE_HEIGHT portion of the output for one (batch, channel) pair
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    int grid_x = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = batch_size * out_channels;  // one block per (batch, channel)
    dim3 grid(grid_x, grid_y, grid_z);

    // Calculate shared memory size: one region for input patch and one for the kernel
    int smem_width  = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (smem_width * smem_height + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_opt_kernel<<<grid, block, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Optimized Depthwise 2D Convolution with Shared Memory and Grid-stride",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
