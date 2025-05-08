#include <torch/extension.h>
#include <vector>

__global__ void conv_transpose2d_kernel(const float* __restrict__ x,
                                        const float* __restrict__ weight,
                                        float* __restrict__ output,
                                        int x_h, int x_w, int w_h, int w_w,
                                        int out_h, int out_w, int stride_h,
                                        int stride_w, int pad_h, int pad_w) {
    extern __shared__ float shared_mem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int out_x = bx * blockDim.x + tx;
    int out_y = by * blockDim.y + ty;

    if (out_x < out_w && out_y < out_h) {
        float value = 0.0f;
        for (int i = 0; i < w_h; ++i) {
            for (int j = 0; j < w_w; ++j) {
                int in_x = (out_x + pad_h) / stride_h - i;
                int in_y = (out_y + pad_w) / stride_w - j;
                if (in_x >= 0 && in_x < x_h && in_y >= 0 && in_y < x_w) {
                    value += x[in_y * x_w + in_x] * weight[i * w_w + j];
                }
            }
        }
        output[out_y * out_w + out_x] = value;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding) {

    const auto x_h = x.size(0);
    const auto x_w = x.size(1);
    const auto w_h = weight.size(0);
    const auto w_w = weight.size(1);
    const auto out_h = (x_h - 1) * stride[0] - 2 * padding[0] + w_h;
    const auto out_w = (x_w - 1) * stride[1] - 2 * padding[1] + w_w;

    auto output = torch::zeros({out_h, out_w}, x.options());

    const dim3 threads(16, 16);
    const dim3 blocks((out_w + threads.x - 1) / threads.x,
                      (out_h + threads.y - 1) / threads.y);

    conv_transpose2d_kernel<<<blocks, threads, 0>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        x_h, x_w, w_h, w_w,
        out_h, out_w, stride[0], stride[1],
        padding[0], padding[1]);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}