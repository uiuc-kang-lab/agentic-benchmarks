#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void depthwise_conv2d_gridstride_warp_kernel(
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
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;
    int oc = bat_oc % out_channels;
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Calculate tile boundaries
    int tile_out_x = blockIdx.x * TILE_WIDTH;
    int tile_out_y = blockIdx.y * TILE_HEIGHT;

    // Calculate input patch dimensions
    int smem_width = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;

    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_weight = shared_mem + smem_height * smem_width;

    // Thread identification within the block and warp
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Load weights cooperatively using warps
    int weight_elements = kernel_size * kernel_size;
    for (int idx = tid; idx < weight_elements; idx += BLOCK_SIZE) {
        s_weight[idx] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) +
            idx
        ];
    }

    // Load input patch using warp-aligned accesses
    int in_start_x = tile_out_x * stride - padding;
    int in_start_y = tile_out_y * stride - padding;
    
    int input_elements = smem_width * smem_height;
    for (int idx = tid; idx < input_elements; idx += BLOCK_SIZE) {
        int local_y = idx / smem_width;
        int local_x = idx % smem_width;
        int global_y = in_start_y + local_y;
        int global_x = in_start_x + local_x;

        float val = 0.0f;
        if (global_y >= 0 && global_y < input_h && 
            global_x >= 0 && global_x < input_w) {
            val = input[
                b * (in_channels * input_h * input_w) +
                in_ch * (input_h * input_w) +
                global_y * input_w +
                global_x
            ];
        }
        s_input[local_y * smem_width + local_x] = val;
    }

    __syncthreads();

    // Process output elements using grid-stride loop with warp-level cooperation
    int tile_elements = TILE_WIDTH * TILE_HEIGHT;
    for (int idx = tid; idx < tile_elements; idx += BLOCK_SIZE) {
        int local_y = idx / TILE_WIDTH;
        int local_x = idx % TILE_WIDTH;
        int out_x = tile_out_x + local_x;
        int out_y = tile_out_y + local_y;

        if (out_x < output_w && out_y < output_h) {
            float sum = 0.0f;
            
            // Compute convolution with warp-aligned memory access pattern
            #pragma unroll
            for (int ky = 0; ky < kernel_size; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int s_y = local_y * stride + ky;
                    int s_x = local_x * stride + kx;
                    float in_val = s_input[s_y * smem_width + s_x];
                    float w_val = s_weight[ky * kernel_size + kx];
                    sum += in_val * w_val;
                }
            }

            if (bias != nullptr) {
                sum += bias[oc];
            }

            // Write output - no atomic needed as each thread writes to a unique location
            output[
                b * (out_channels * output_h * output_w) +
                oc * (output_h * output_w) +
                out_y * output_w +
                out_x
            ] = sum;
        }
    }
}

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

    dim3 grid(
        (output_w + TILE_WIDTH - 1) / TILE_WIDTH,
        (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch_size * out_channels
    );
    
    dim3 block(BLOCK_SIZE);

    int smem_width = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;
    size_t shared_mem_size = (smem_width * smem_height + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_gridstride_warp_kernel<<<grid, block, shared_mem_size>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Warp-Level Optimization (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}