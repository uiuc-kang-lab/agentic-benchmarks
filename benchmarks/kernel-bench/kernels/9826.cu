#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

__global__ void dw2d_index_optimized_kernel(
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
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_weight = shared_mem + (TILE_HEIGHT * TILE_WIDTH);

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_HEIGHT + ty;
    int batch = blockIdx.z / out_channels;
    int oc = blockIdx.z % out_channels;

    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    int smem_rows = (TILE_HEIGHT - 1) * stride + kernel_size;
    int smem_cols = (TILE_WIDTH - 1) * stride + kernel_size;

    int in_start_y = row * stride - padding;
    int in_start_x = col * stride - padding;

    int num_weight = kernel_size * kernel_size;
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < num_weight; i += blockDim.x * blockDim.y) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            weight_ch * (kernel_size * kernel_size) +
            i
        ];
    }

    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < smem_rows * smem_cols; i += blockDim.x * blockDim.y) {
        int r = i / smem_cols;
        int c = i % smem_cols;
        float val = 0.0f;
        int global_y = in_start_y + r;
        int global_x = in_start_x + c;
        if (global_y >= 0 && global_y < input_h && global_x >= 0 && global_x < input_w) {
            int input_idx = batch * (in_channels * input_h * input_w) +
                            in_ch * (input_h * input_w) +
                            global_y * input_w + global_x;
            val = input[input_idx];
        }
        s_input[r * smem_cols + c] = val;
    }

    __syncthreads();

    if (ty < TILE_HEIGHT && tx < TILE_WIDTH && row < output_h && col < output_w) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int ix = tx * stride + kx;
                int iy = ty * stride + ky;
                sum += s_input[iy * smem_cols + ix] * s_weight[ky * kernel_size + kx];
            }
        }

        if (bias != nullptr) {
            sum += bias[oc];
        }

        output[batch * out_channels * output_h * output_w + oc * output_h * output_w + row * output_w + col] = sum;
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

    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    int grid_x = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = batch_size * out_channels;
    dim3 grid(grid_x, grid_y, grid_z);

    int smem_size = (TILE_HEIGHT * TILE_WIDTH + kernel_size * kernel_size) * sizeof(float);

    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;

    dw2d_index_optimized_kernel<<<grid, block, smem_size>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Shared Memory and Optimized Indexing (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}