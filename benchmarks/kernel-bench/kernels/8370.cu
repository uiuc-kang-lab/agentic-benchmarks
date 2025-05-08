#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int x_height, int x_width,
    int weight_height, int weight_width,
    int out_height, int out_width,
    int stride_h, int stride_w,
    int padding_h, int padding_w) {

    extern __shared__ float shared_data[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int out_x = bx + tx;
    int out_y = by + ty;

    if (out_x < out_width && out_y < out_height) {
        float value = 0.0f;
        for (int i = 0; i < weight_height; ++i) {
            for (int j = 0; j < weight_width; ++j) {
                int x_x = out_x - i * stride_w + padding_w;
                int x_y = out_y - j * stride_h + padding_h;
                if (x_x >= 0 && x_x < x_width && x_y >= 0 && x_y < x_height) {
                    int x_index = x_y * x_width + x_x;
                    int weight_index = i * weight_width + j;
                    value += x[x_index] * weight[weight_index];
                }
            }
        }
        shared_data[ty * blockDim.x + tx] = value;
        __syncthreads();

        // Warp-level reduction
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }

        if (tx % warpSize == 0) {
            atomicAdd(&output[out_y * out_width + out_x], value);
        }
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding) {

    const auto x_height = x.size(2);
    const auto x_width = x.size(3);
    const auto weight_height = weight.size(2);
    const auto weight_width = weight.size(3);

    const auto out_height = (x_height - 1) * stride[0] - 2 * padding[0] + weight_height;
    const auto out_width = (x_width - 1) * stride[1] - 2 * padding[1] + weight_width;

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::zeros({x.size(0), weight.size(1), out_height, out_width}, options);

    const dim3 threads(16, 16);
    const dim3 blocks((out_width + threads.x - 1) / threads.x,
                      (out_height + threads.y - 1) / threads.y);

    size_t shared_memory_size = threads.x * threads.y * sizeof(float);

    conv_transpose2d_kernel<<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        x_height, x_width,
        weight_height, weight_width,
        out_height, out_width,
        stride[0], stride[1],
        padding[0], padding[1]);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}