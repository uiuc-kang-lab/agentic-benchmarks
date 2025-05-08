#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define BLOCK_SIZE 128  // Optimized block size for H100 GPU

__global__ void depthwise_conv2d_tuned_blocksize_kernel(
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
    int in_start_x = tile_out_x * stride - padding;
    int in_start_y = tile_out_y * stride - padding;

    // Calculate shared memory dimensions
    int smem_width = (TILE_WIDTH - 1) * stride + kernel_size;
    int smem_height = (TILE_HEIGHT - 1) * stride + kernel_size;

    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_weight = shared_mem + smem_height * smem_width;

    int tid = threadIdx.x;

    // Load weights into shared memory - optimized for 128 threads
    int total_weight = kernel_size * kernel_size;
    for (int i = tid; i < total_weight; i += BLOCK_SIZE) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) + i
        ];
    }

    // Load input patch into shared memory - optimized for 128 threads
    int total_input = smem_height * smem_width;
    #pragma unroll 4
    for (int i = tid; i < total_input; i += BLOCK_SIZE) {
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

    // Process output elements - optimized for 128 threads
    int tile_elements = TILE_WIDTH * TILE_HEIGHT;
    #pragma unroll 4
    for (int i = tid; i < tile_elements; i += BLOCK_SIZE) {
        int local_y = i / TILE_WIDTH;
        int local_x = i % TILE_WIDTH;
        int out_x = tile_out_x + local_x;
        int out_y = tile_out_y + local_y;

        if (out_x < output_w && out_y < output_h) {
            float sum = 0.0f;

            if (kernel_size == 3) {
                // Manually unrolled 3x3 convolution
                int base_y = local_y * stride;
                int base_x = local_x * stride;
                
                sum += s_input[base_y * smem_width + base_x] * s_weight[0];
                sum += s_input[base_y * smem_width + (base_x + 1)] * s_weight[1];
                sum += s_input[base_y * smem_width + (base_x + 2)] * s_weight[2];
                
                base_y = (local_y * stride + 1);
                sum += s_input[base_y * smem_width + base_x] * s_weight[3];
                sum += s_input[base_y * smem_width + (base_x + 1)] * s_weight[4];
                sum += s_input[base_y * smem_width + (base_x + 2)] * s_weight[5];
                
                base_y = (local_y * stride + 2);
                sum += s_input[base_y * smem_width + base_x] * s_weight[6];
                sum += s_input[base_y * smem_width + (base_x + 1)] * s_weight[7];
                sum += s_input[base_y * smem_width + (base_x + 2)] * s_weight[8];
            } else {
                #pragma unroll
                for (int ky = 0; ky < kernel_size; ++ky) {
                    #pragma unroll
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int s_y = local_y * stride + ky;
                        int s_x = local_x * stride + kx;
                        sum += s_input[s_y * smem_width + s_x] * 
                               s_weight[ky * kernel_size + kx];
                    }
                }
            }

            if (bias != nullptr) {
                sum += bias[oc];
            }

            output[b * (out_channels * output_h * output_w) +
                   oc * (output_h * output_w) +
                   out_y * output_w + out_x] = sum;
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
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Input and weight must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D tensor");

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
    size_t shared_mem_bytes = (smem_width * smem_height + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_tuned_blocksize_kernel<<<grid, block, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Tuned Block Size (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}