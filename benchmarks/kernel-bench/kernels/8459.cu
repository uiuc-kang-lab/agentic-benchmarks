#include <torch/extension.h>
#include <vector>

// CUDA kernel for 2D transposed convolution
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int x_height, int x_width,
    int weight_height, int weight_width,
    int out_height, int out_width,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups) {

    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_y < out_height && out_x < out_width) {
        float sum = 0.0f;
        for (int g = 0; g < groups; ++g) {
            for (int ky = 0; ky < weight_height; ++ky) {
                for (int kx = 0; kx < weight_width; ++kx) {
                    int in_y = out_y - ky * dilation_h + pad_h;
                    int in_x = out_x - kx * dilation_w + pad_w;
                    if (in_y % stride_h == 0 && in_x % stride_w == 0) {
                        in_y /= stride_h;
                        in_x /= stride_w;
                        if (in_y >= 0 && in_y < x_height && in_x >= 0 && in_x < x_width) {
                            sum += x[((g * x_height + in_y) * x_width + in_x)] *
                                   weight[((g * weight_height + ky) * weight_width + kx)];
                        }
                    }
                }
            }
        }
        output[out_y * out_width + out_x] = sum;
    }
}

// Host function to launch the CUDA kernel
void conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    const int x_height = x.size(2);
    const int x_width = x.size(3);
    const int weight_height = weight.size(2);
    const int weight_width = weight.size(3);
    const int out_height = output.size(2);
    const int out_width = output.size(3);

    const dim3 threads(16, 16);
    const dim3 blocks((out_width + threads.x - 1) / threads.x,
                      (out_height + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            x_height, x_width,
            weight_height, weight_width,
            out_height, out_width,
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            groups);
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}