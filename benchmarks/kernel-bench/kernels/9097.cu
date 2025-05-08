#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__global__ void conv_transpose2d_kernel(
    const float *x,
    const float *weight,
    const float *bias,
    float *output,
    int x_height,
    int x_width,
    int weight_height,
    int weight_width,
    int out_height,
    int out_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width) {

    extern __shared__ float shared_mem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int row = bidy * stride_height + ty - padding_height;
    int col = bidx * stride_width + tx - padding_width;

    if (row < 0 || row >= out_height || col < 0 || col >= out_width) return;
    
    float result = 0.0f;

    for (int i = 0; i < weight_height; ++i) {
        for (int j = 0; j < weight_width; ++j) {
            int x_row = row - i;
            int x_col = col - j;
            if (x_row >= 0 && x_row < x_height && x_col >= 0 && x_col < x_width) {
                int x_index = x_row * x_width + x_col;
                int w_index = i * weight_width + j;
                result += x[x_index] * weight[w_index];
            }
        }
    }

    shared_mem[ty * blockDim.x + tx] = result;
    __syncthreads();

    // Warp-level reduction
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        result += __shfl_down_sync(0xffffffff, result, offset);
    }

    if (tx == 0 && ty == 0) {
        if (bias != nullptr) {
            result += bias[bidy * gridDim.x + bidx];
        }
        output[row * out_width + col] = result;
    }
}

void conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding)
{
    const dim3 threads(16, 16);
    const dim3 blocks((output.size(1) + threads.x - 1) / threads.x,
                      (output.size(0) + threads.y - 1) / threads.y);
    int shared_mem_size = threads.x * threads.y * sizeof(float);

    conv_transpose2d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        x.size(0),
        x.size(1),
        weight.size(0),
        weight.size(1),
        output.size(0),
        output.size(1),
        stride[0],
        stride[1],
        padding[0],
        padding[1]);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_cuda", &conv_transpose2d_cuda, "Conv Transpose 2D with shared memory and warp reduction",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("output"),
          py::arg("stride"),
          py::arg("padding"));
}