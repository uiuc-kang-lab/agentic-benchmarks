#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions and block parameters
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Kernel that uses shared memory for input and weight, and warp-level reduction using __shfl_down_sync()
__global__ void depthwise_conv2d_warpred_kernel(
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
    // Identify batch and output channel
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Compute tile's starting output coordinates
    int tile_out_x = blockIdx.x * TILE_WIDTH;
    int tile_out_y = blockIdx.y * TILE_HEIGHT;

    // Compute the starting input coordinate for the tile
    int in_start_x = tile_out_x * stride - padding;
    int in_start_y = tile_out_y * stride - padding;

    // Compute shared memory dimensions for the input patch
    int smem_width  = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;

    // Declare shared memory: first region for input patch, then for weight kernel
    extern __shared__ float shared_mem[];
    float* s_input  = shared_mem;                           // size: smem_width * smem_height
    float* s_weight = shared_mem + (smem_width * smem_height); // size: kernel_size * kernel_size

    // Load weight kernel into shared memory
    int tid = threadIdx.x;
    int total_weight = kernel_size * kernel_size;
    for (int i = tid; i < total_weight; i += blockDim.x) {
        int weight_idx = in_ch * (channels_per_group * total_weight) +
                         weight_ch * total_weight + i;
        s_weight[i] = weight[weight_idx];
    }

    // Load the input patch into shared memory
    int total_patch = smem_width * smem_height;
    for (int i = tid; i < total_patch; i += blockDim.x) {
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

    // Each block computes a tile of output elements. We'll assign groups of threads (warps) to each output element.
    int tile_elems = TILE_WIDTH * TILE_HEIGHT; // Number of output elements in this tile
    int num_warps = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Loop over output elements in the tile assigned to this warp
    for (int idx = warp_id; idx < tile_elems; idx += num_warps) {
        int local_y = idx / TILE_WIDTH;
        int local_x = idx % TILE_WIDTH;
        int out_x = tile_out_x + local_x;
        int out_y = tile_out_y + local_y;

        if (out_x < output_w && out_y < output_h) {
            float partial_sum = 0.0f;
            int total_kernel_elems = kernel_size * kernel_size;
            // Each thread in the warp processes a subset of the kernel multiplications
            for (int k = lane; k < total_kernel_elems; k += WARP_SIZE) {
                int ky = k / kernel_size;
                int kx = k % kernel_size;
                int s_y = local_y * stride + ky;
                int s_x = local_x * stride + kx;
                float in_val = s_input[s_y * smem_width + s_x];
                float w_val = s_weight[k];
                partial_sum += in_val * w_val;
            }
            
            // Warp-level reduction using __shfl_down_sync
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
            }
            
            // Lane 0 of the warp writes the final sum for this output element
            if (lane == 0) {
                if (bias != nullptr) {
                    partial_sum += bias[oc];
                }
                int out_idx = b * (out_channels * output_h * output_w) +
                              oc * (output_h * output_w) +
                              out_y * output_w + out_x;
                output[out_idx] = partial_sum;
            }
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

    // Grid dimensions: each block processes a tile for one (batch, channel) pair
    int grid_x = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = batch_size * out_channels;
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(BLOCK_SIZE);

    // Compute shared memory size: input patch + weight kernel
    int smem_width = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;
    size_t shared_mem_bytes = (smem_width * smem_height + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_warpred_kernel<<<grid, block, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Warp-level Reduction (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
