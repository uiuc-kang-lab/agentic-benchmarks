#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

__device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int x_height, int x_width,
    int weight_height, int weight_width,
    int out_height, int out_width,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_width && out_y < out_height) {
        float sum = 0.0f;
        for (int k_y = 0; k_y < weight_height; ++k_y) {
            for (int k_x = 0; k_x < weight_width; ++k_x) {
                int in_x = out_x - k_x + pad_w;
                int in_y = out_y - k_y + pad_h;
                if (in_x % stride_w == 0 && in_y % stride_h == 0) {
                    in_x /= stride_w;
                    in_y /= stride_h;
                    if (in_x >= 0 && in_x < x_width && in_y >= 0 && in_y < x_height) {
                        sum += x[in_y * x_width + in_x] * weight[k_y * weight_width + k_x];
                    }
                }
            }
        }
        sum = warpReduceSum(sum);
        if (threadIdx.x % warpSize == 0) {
            output[out_y * out_width + out_x] = sum;
        }
    }
}

void conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding) {

    const int x_height = x.size(0);
    const int x_width = x.size(1);
    const int weight_height = weight.size(0);
    const int weight_width = weight.size(1);
    const int out_height = output.size(0);
    const int out_width = output.size(1);

    const dim3 threadsPerBlock(32, 32);
    const dim3 numBlocks((out_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (out_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conv_transpose2d_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        x_height, x_width,
        weight_height, weight_width,
        out_height, out_width,
        stride[0], stride[1],
        padding[0], padding[1]);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}