#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block dimensions: 2D block mapping to output tile
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

// Optimized kernel using 2D thread blocks for direct mapping to output spatial dimensions
__global__ void depthwise_conv2d_2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    // Each block is responsible for one (batch, output channel) pair
    int b_oc = blockIdx.z; // b_oc ranges from 0 to batch_size * out_channels - 1
    int b = b_oc / out_channels;
    int oc = b_oc % out_channels;

    // Map output channel to corresponding input channel and weight subgroup
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Determine output coordinate within the feature map using 2D block indexing
    int out_x = blockIdx.x * BLOCK_WIDTH + threadIdx.x; // column index
    int out_y = blockIdx.y * BLOCK_HEIGHT + threadIdx.y;  // row index

    // Dimensions of shared memory tile for the input patch
    int smem_width  = (BLOCK_WIDTH - 1) * stride + kernel_size;
    int smem_height = (BLOCK_HEIGHT - 1) * stride + kernel_size;

    // Allocate shared memory: first for weight, then for input patch
    extern __shared__ float shared_mem[];
    float* s_weight = shared_mem; // size: kernel_size * kernel_size
    float* s_input = shared_mem + kernel_size * kernel_size; // size: smem_width * smem_height

    // Load the weight kernel into shared memory cooperatively
    int tid = threadIdx.y * BLOCK_WIDTH + threadIdx.x;
    int total_weight = kernel_size * kernel_size;
    for (int i = tid; i < total_weight; i += BLOCK_WIDTH * BLOCK_HEIGHT) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) + i
        ];
    }

    // Compute the top-left corner of the input patch corresponding to this output tile
    int in_x0 = blockIdx.x * BLOCK_WIDTH * stride - padding;
    int in_y0 = blockIdx.y * BLOCK_HEIGHT * stride - padding;

    // Load the input patch into shared memory using 2D cooperative loading
    for (int j = threadIdx.y; j < smem_height; j += BLOCK_HEIGHT) {
        for (int i = threadIdx.x; i < smem_width; i += BLOCK_WIDTH) {
            int global_x = in_x0 + i;
            int global_y = in_y0 + j;
            float val = 0.0f;
            if (global_y >= 0 && global_y < input_h && global_x >= 0 && global_x < input_w) {
                int global_index = b * (in_channels * input_h * input_w) +
                                   in_ch * (input_h * input_w) +
                                   global_y * input_w + global_x;
                val = input[global_index];
            }
            s_input[j * smem_width + i] = val;
        }
    }

    __syncthreads();

    // Only compute if the output coordinate is within bounds
    if (out_x < out_w && out_y < out_h) {
        float sum = 0.0f;
        // Compute the starting position in shared memory for this output element
        int s_x = threadIdx.x * stride;
        int s_y = threadIdx.y * stride;
        
        // Perform the convolution using the weights loaded in shared memory
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                float inp = s_input[(s_y + ky) * smem_width + (s_x + kx)];
                float wt = s_weight[ky * kernel_size + kx];
                sum += inp * wt;
            }
        }

        // Apply bias if provided
        if (bias != nullptr) {
            sum += bias[oc];
        }

        int out_index = b * (out_channels * out_h * out_w) +
                        oc * (out_h * out_w) +
                        out_y * out_w + out_x;
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

    int out_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

    // Set up grid dimensions: Each block processes a tile of size BLOCK_WIDTH x BLOCK_HEIGHT for one (batch, channel) pair
    int grid_x = (out_w + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    int grid_y = (out_h + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
    int grid_z = batch_size * out_channels;
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);

    // Calculate shared memory size: weight + input patch
    int smem_width  = (BLOCK_WIDTH - 1) * stride + kernel_size;
    int smem_height = (BLOCK_HEIGHT - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (kernel_size * kernel_size + smem_width * smem_height) * sizeof(float);

    const float* bias_ptr = (bias.has_value() ? bias->data_ptr<float>() : nullptr);

    depthwise_conv2d_2d_shared_kernel<<<grid, block, shared_mem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        out_h,
        out_w,
        kernel_size,
        stride,
        padding,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution with 2D Block Indexing and Shared Memory (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
