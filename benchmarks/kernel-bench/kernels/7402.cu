#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward function definition
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t x_h, int64_t x_w, int64_t weight_h, int64_t weight_w, 
    int64_t out_h, int64_t out_w, int64_t stride, int64_t padding, int64_t output_padding, int64_t groups) {

    // Calculate output index
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_w && out_y < out_h) {
        float value = 0.0f;

        for (int c = 0; c < groups; ++c) {
            for (int ky = 0; ky < weight_h; ++ky) {
                for (int kx = 0; kx < weight_w; ++kx) {
                    int in_x = out_x - kx + padding - output_padding;
                    int in_y = out_y - ky + padding - output_padding;

                    if (in_x % stride == 0 && in_y % stride == 0) {
                        in_x /= stride;
                        in_y /= stride;

                        if (in_x >= 0 && in_x < x_w && in_y >= 0 && in_y < x_h) {
                            value += __ldg(&x[(c * x_h + in_y) * x_w + in_x]) * __ldg(&weight[(c * weight_h + ky) * weight_w + kx]);
                        }
                    }
                }
            }
        }

        output[out_y * out_w + out_x] = value;
    }
}

// Forward function definition
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Ensure inputs are on CUDA and contiguous
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    auto x_sizes = x.sizes();
    auto weight_sizes = weight.sizes();

    int64_t x_h = x_sizes[2];
    int64_t x_w = x_sizes[3];
    int64_t weight_h = weight_sizes[2];
    int64_t weight_w = weight_sizes[3];
    int64_t out_h = (x_h - 1) * stride - 2 * padding + weight_h + output_padding;
    int64_t out_w = (x_w - 1) * stride - 2 * padding + weight_w + output_padding;

    auto output = torch::empty({x_sizes[0], groups, out_h, out_w}, x.options());

    dim3 blockDim(16, 16);
    dim3 gridDim((out_w + blockDim.x - 1) / blockDim.x, (out_h + blockDim.y - 1) / blockDim.y);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_forward", ([&] {
        conv_transpose2d_kernel<<<gridDim, blockDim>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            x_h, x_w, weight_h, weight_w, out_h, out_w, stride, padding, output_padding, groups);
    }));

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA)");
}