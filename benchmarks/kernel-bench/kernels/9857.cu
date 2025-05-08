#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions for the output
#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define BLOCK_SIZE 256

// Device function to load the weight kernel into shared memory
__device__ inline void load_weights(const float* __restrict__ weight, float* __restrict__ s_weight,
                                       int in_ch, int weight_ch, int kernel_size, int channels_per_group, int blockSize) {
    int tid = threadIdx.x;
    int total_weight = kernel_size * kernel_size;
    for (int i = tid; i < total_weight; i += blockSize) {
        s_weight[i] = weight[in_ch * (channels_per_group * kernel_size * kernel_size) +
                             weight_ch * (kernel_size * kernel_size) + i];
    }
}

// Device function to load the input patch into shared memory
__device__ inline void load_input_patch(const float* __restrict__ input, float* __restrict__ s_input,
                                          int b, int in_ch, int input_h, int input_w,
                                          int in_start_y, int in_start_x, int smem_width, int smem_height,
                                          int in_channels, int blockSize) {
    int tid = threadIdx.x;
    int total_input = smem_width * smem_height;
    for (int i = tid; i < total_input; i += blockSize) {
        int r = i / smem_width;
        int c = i % smem_width;
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
}

// Device function to compute convolution for a single output element
__device__ inline float compute_convolution(const float* __restrict__ s_input, const float* __restrict__ s_weight,
                                               int local_y, int local_x, int stride, int kernel_size, int smem_width) {
    float sum = 0.0f;
    if (kernel_size == 3) {
        // Manual unrolling for 3x3 kernel
        sum += s_input[(local_y * stride) * smem_width + (local_x * stride)] * s_weight[0];
        sum += s_input[(local_y * stride) * smem_width + (local_x * stride + 1)] * s_weight[1];
        sum += s_input[(local_y * stride) * smem_width + (local_x * stride + 2)] * s_weight[2];

        sum += s_input[((local_y * stride) + 1) * smem_width + (local_x * stride)] * s_weight[3];
        sum += s_input[((local_y * stride) + 1) * smem_width + (local_x * stride + 1)] * s_weight[4];
        sum += s_input[((local_y * stride) + 1) * smem_width + (local_x * stride + 2)] * s_weight[5];

        sum += s_input[((local_y * stride) + 2) * smem_width + (local_x * stride)] * s_weight[6];
        sum += s_input[((local_y * stride) + 2) * smem_width + (local_x * stride + 1)] * s_weight[7];
        sum += s_input[((local_y * stride) + 2) * smem_width + (local_x * stride + 2)] * s_weight[8];
    } else {
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                int s_y = local_y * stride + ky;
                int s_x = local_x * stride + kx;
                sum += s_input[s_y * smem_width + s_x] * s_weight[ky * kernel_size + kx];
            }
        }
    }
    return sum;
}

// Main kernel: modularized depthwise convolution using device helper functions
__global__ void depthwise_conv2d_modular_kernel(
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
    // Each block processes one (batch, output channel) pair
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Determine the starting coordinates of the output tile
    int tile_out_x = blockIdx.x * TILE_WIDTH;
    int tile_out_y = blockIdx.y * TILE_HEIGHT;

    // Compute the corresponding top-left input coordinate for this output tile
    int in_start_x = tile_out_x * stride - padding;
    int in_start_y = tile_out_y * stride - padding;

    // Dimensions of the shared memory input patch
    int smem_width  = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;

    extern __shared__ float shared_mem[];
    float* s_input  = shared_mem;                          // size: smem_width * smem_height
    float* s_weight = shared_mem + smem_width * smem_height; // size: kernel_size * kernel_size

    int blockSize = blockDim.x;  // assume 1D block
    
    // Load weight and input patch into shared memory using modular device functions
    load_weights(weight, s_weight, in_ch, weight_ch, kernel_size, channels_per_group, blockSize);
    load_input_patch(input, s_input, b, in_ch, input_h, input_w, in_start_y, in_start_x, smem_width, smem_height, in_channels, blockSize);

    __syncthreads();

    // Each thread computes multiple output elements in the tile using grid-stride loop
    int tile_area = TILE_WIDTH * TILE_HEIGHT;
    for (int i = threadIdx.x; i < tile_area; i += blockSize) {
        int local_y = i / TILE_WIDTH;
        int local_x = i % TILE_WIDTH;
        int out_x = tile_out_x + local_x;
        int out_y = tile_out_y + local_y;

        if (out_x < output_w && out_y < output_h) {
            float sum = compute_convolution(s_input, s_weight, local_y, local_x, stride, kernel_size, smem_width);
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

// Forward function callable from Python using pybind11
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

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    int grid_x = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = batch_size * out_channels;
    dim3 grid(grid_x, grid_y, grid_z);

    dim3 block(BLOCK_SIZE);

    int smem_width  = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (smem_width * smem_height + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_modular_kernel<<<grid, block, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Modular Depthwise 2D Convolution with Device Functions (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
