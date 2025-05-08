#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

__global__ void conv_transpose2d_kernel_shared(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {

    __shared__ float shared_weight[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_input[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    const int batch_id = blockIdx.z / out_channels;
    const int out_ch = blockIdx.z % out_channels;

    const int out_x = bx + tx;
    const int out_y = by + ty;

    float sum = 0.0f;

    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int i = ty; i < kernel_height; i += TILE_SIZE) {
            for (int j = tx; j < kernel_width; j += TILE_SIZE) {
                if (i < kernel_height && j < kernel_width) {
                    shared_weight[i][j] = weight[
                        in_ch * out_channels * kernel_height * kernel_width +
                        out_ch * kernel_height * kernel_width +
                        i * kernel_width + j
                    ];
                }
            }
        }
        __syncthreads();

        if (out_x < output_width && out_y < output_height) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int in_x = out_x + pad_w - kw;
                    int in_y = out_y + pad_h - kh;

                    if (in_x % stride_w == 0 && in_y % stride_h == 0) {
                        in_x /= stride_w;
                        in_y /= stride_h;

                        if (in_x >= 0 && in_x < input_width && 
                            in_y >= 0 && in_y < input_height) {
                            
                            shared_input[ty][tx] = input[
                                batch_id * in_channels * input_height * input_width +
                                in_ch * input_height * input_width +
                                in_y * input_width + in_x
                            ];
                            __syncthreads();

                            sum += shared_input[ty][tx] * shared_weight[kh][kw];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (out_x < output_width && out_y < output_height) {
        if (bias != nullptr) {
            sum += bias[out_ch];
        }

        output[
            batch_id * out_channels * output_height * output_width +
            out_ch * output_height * output_width +
            out_y * output_width + out_x
        ] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + 
                             kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + 
                            kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                             x.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * out_channels
    );

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv_transpose2d_kernel_shared<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Shared memory ConvTranspose2D forward (CUDA)");
}