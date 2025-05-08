#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for the output tile
#define TILE_WIDTH 32
#define TILE_HEIGHT 32

// This kernel uses manual loop unrolling for kernel size 3x3 and #pragma unroll for other sizes

__global__ void depthwise_conv2d_unroll_gridstride_shared_kernel(
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
    // Each block corresponds to one (batch, output channel) pair
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;

    // Map output channel back to input channel and its corresponding weight subgroup
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Determine the starting coordinates of the output tile processed by this block
    int tile_out_x = blockIdx.x * TILE_WIDTH;
    int tile_out_y = blockIdx.y * TILE_HEIGHT;

    // Compute the corresponding top-left input coordinate for this output tile
    int in_start_x = tile_out_x * stride - padding;
    int in_start_y = tile_out_y * stride - padding;

    // Determine the dimensions of the shared memory patch needed
    int smem_width  = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;

    // Allocate shared memory: first for the input patch, then for the weight kernel
    extern __shared__ float shared_mem[];
    float* s_input  = shared_mem;                          // Size: smem_height * smem_width
    float* s_weight = shared_mem + smem_height * smem_width; // Size: kernel_size * kernel_size

    // Use a flat 1D block of threads to cooperatively load data
    int tid = threadIdx.x;
    int blockSize = blockDim.x;  // Block is launched as 1D (e.g. 256 threads)

    // Load the weight kernel into shared memory
    int total_weight = kernel_size * kernel_size;
    for (int i = tid; i < total_weight; i += blockSize) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) + i
        ];
    }

    // Load the required input patch into shared memory using a grid-stride loop
    int total_input = smem_height * smem_width;
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

    __syncthreads();

    // Total number of output elements in the tile
    int tile_area = TILE_WIDTH * TILE_HEIGHT;
    // Each thread using grid-stride loop computes multiple outputs within the tile
    for (int i = tid; i < tile_area; i += blockSize) {
        int local_y = i / TILE_WIDTH;
        int local_x = i % TILE_WIDTH;
        int out_x = tile_out_x + local_x;
        int out_y = tile_out_y + local_y;
        
        if (out_x < output_w && out_y < output_h) {
            float sum = 0.0f;
            // Compute convolution sum by accessing shared memory (input patch) with given stride
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
                // Use #pragma unroll for other kernel sizes
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

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Grid dimensions: each block covers a TILE_WIDTH x TILE_HEIGHT output tile for one (batch, channel) pair
    int grid_x = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = batch_size * out_channels;
    dim3 grid(grid_x, grid_y, grid_z);

    // Launch a 1D block of threads evenly distributing the tile workload
    int blockSize = 256;
    dim3 block(blockSize);

    // Calculate required shared memory: for the input patch plus the kernel weights
    int smem_width  = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (smem_width * smem_height + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_unroll_gridstride_shared_kernel<<<grid, block, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Manual Unrolling and Grid-Stride (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
