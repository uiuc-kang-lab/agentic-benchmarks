#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int TILE_SIZE = 16;
const int KERNEL_MAX = 7;

__global__ void conv2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width) {

    __shared__ float sh_input[TILE_SIZE + KERNEL_MAX][TILE_SIZE + KERNEL_MAX];

    const int in_row_start = out_row * stride - padding;
    const int in_col_start = out_col * stride - padding;

    for (int i = tidy; i < TILE_SIZE + kernel_size - 1; i += blockDim.y) {
        for (int j = tidx; j < TILE_SIZE + kernel_size - 1; j += blockDim.x) {
            int in_row = in_row_start + i;
            int in_col = in_col_start + j;
            if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width) {
                sh_input[i][j] = input[((b * in_channels + c_in) * in_height + in_row) * in_width + in_col];
            } else {
                sh_input[i][j] = 0.0f;
            }
        }
    }
    __syncthreads();
    
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int out_row = blockIdx.y * TILE_SIZE + tidy;
    const int out_col = blockIdx.x * TILE_SIZE + tidx;
    const int batch_channel = blockIdx.z;
    const int b = batch_channel / out_channels;
    const int c_out = batch_channel % out_channels;
    
    if (b >= batch_size || c_out >= out_channels) return;

    float sum = 0.0f;
    
    const int in_row_start = out_row * stride - padding;
    const int in_col_start = out_col * stride - padding;
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        #pragma unroll
        for (int k_row = 0; k_row < kernel_size; ++k_row) {
            #pragma unroll
            for (int k_col = 0; k_col < kernel_size; ++k_col) {
                const int in_row = in_row_start + k_row * dilation;
                const int in_col = in_col_start + k_col * dilation;
                
                if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width) {
                    const float val = input[((b * in_channels + c_in) * in_height + in_row) * in_width + in_col];
                    const float w = weight[((c_out * in_channels + c_in) * kernel_size + k_row) * kernel_size + k_col];
                    sum += val * w;
                }
            }
        }
    }
    
    if (out_row < out_height && out_col < out_width) {
        const int out_idx = ((b * out_channels + c_out) * out_height + out_row) * out_width + out_col;
        output[out_idx] = sum + (bias ? bias[c_out] : 0.0f);
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) CHECK_INPUT(bias.value());

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);

    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch_size * out_channels
    );

    conv2d_shared_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        out_height,
        out_width);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 2D convolution with shared memory tiling");
}