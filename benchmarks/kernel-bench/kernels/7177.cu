#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 16
#define BLOCK_SIZE 16

__global__ void conv2d_balanced_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];
    
    const int batch_idx = blockIdx.x;
    const int channel_idx = blockIdx.y;
    const int tile_row = blockIdx.z / ((out_width + TILE_SIZE - 1) / TILE_SIZE);
    const int tile_col = blockIdx.z % ((out_width + TILE_SIZE - 1) / TILE_SIZE);
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    const int out_y = tile_row * TILE_SIZE + thread_row;
    const int out_x = tile_col * TILE_SIZE + thread_col;

    if (out_y >= out_height || out_x >= out_width) return;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        const int in_y_base = out_y * stride - padding;
        const int in_x_base = out_x * stride - padding;

        #pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int in_y = in_y_base + ky * dilation;
                const int in_x = in_x_base + kx * dilation;

                float in_val = 0.0f;
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    in_val = input[
                        batch_idx * in_channels * in_height * in_width +
                        ic * in_height * in_width +
                        in_y * in_width +
                        in_x
                    ];
                }

                const float weight_val = weight[
                    channel_idx * in_channels * kernel_size * kernel_size +
                    ic * kernel_size * kernel_size +
                    ky * kernel_size +
                    kx
                ];

                sum += in_val * weight_val;
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[channel_idx];
    }

    const int out_idx = 
        batch_idx * out_channels * out_height * out_width +
        channel_idx * out_height * out_width +
        out_y * out_width +
        out_x;
    
    output[out_idx] = sum;
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
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    TORCH_CHECK(groups == 1, "Only groups==1 is supported");

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        batch,
        out_channels,
        ((out_height + TILE_SIZE - 1) / TILE_SIZE) * ((out_width + TILE_SIZE - 1) / TILE_SIZE)
    );

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv2d_balanced_kernel<<<blocks, threads>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced workload CUDA convolution implementation");
}