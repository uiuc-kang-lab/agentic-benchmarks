#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 16
#define BLOCK_SIZE 16

template<int TILE_WIDTH>
__global__ void conv2d_shared_memory_kernel(
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

    __shared__ float shared_input[TILE_WIDTH + 2][TILE_WIDTH + 2];
    __shared__ float shared_weight[TILE_WIDTH][TILE_WIDTH];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_WIDTH;
    const int by = blockIdx.y * TILE_WIDTH;
    const int batch_idx = blockIdx.z / out_channels;
    const int oc = blockIdx.z % out_channels;

    const int out_x = bx + tx;
    const int out_y = by + ty;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int i = ty; i < TILE_WIDTH + 2; i += BLOCK_SIZE) {
            for (int j = tx; j < TILE_WIDTH + 2; j += BLOCK_SIZE) {
                int in_y = by * stride + i - padding;
                int in_x = bx * stride + j - padding;

                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    shared_input[i][j] = input[
                        batch_idx * in_channels * in_height * in_width +
                        ic * in_height * in_width +
                        in_y * in_width +
                        in_x
                    ];
                } else {
                    shared_input[i][j] = 0.0f;
                }
            }
        }

        for (int i = ty; i < kernel_size; i += BLOCK_SIZE) {
            for (int j = tx; j < kernel_size; j += BLOCK_SIZE) {
                if (i < kernel_size && j < kernel_size) {
                    shared_weight[i][j] = weight[
                        oc * in_channels * kernel_size * kernel_size +
                        ic * kernel_size * kernel_size +
                        i * kernel_size +
                        j
                    ];
                }
            }
        }

        __syncthreads();

        if (out_x < out_width && out_y < out_height) {
            #pragma unroll
            for (int ky = 0; ky < kernel_size; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_y = ty * stride + ky * dilation;
                    int in_x = tx * stride + kx * dilation;
                    sum += shared_input[in_y][in_x] * shared_weight[ky][kx];
                }
            }
        }

        __syncthreads();
    }

    if (out_x < out_width && out_y < out_height) {
        if (bias != nullptr) {
            sum += bias[oc];
        }

        output[
            batch_idx * out_channels * out_height * out_width +
            oc * out_height * out_width +
            out_y * out_width +
            out_x
        ] = sum;
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
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch * out_channels
    );

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv2d_shared_memory_kernel<TILE_SIZE><<<blocks, threads>>>(
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
    m.def("forward", &forward, "Shared memory optimized CUDA convolution implementation");
}