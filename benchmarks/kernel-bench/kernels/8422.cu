#include <torch/extension.h>
#include <vector>

__global__ void conv_transpose2d_kernel(float *input, float *kernel, float *output,
                                        int input_w, int input_h, int kernel_w, int kernel_h,
                                        int stride_w, int stride_h, int output_w, int output_h) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < output_w && out_y < output_h) {
        float sum = 0.0f;
        for (int kx = 0; kx < kernel_w; ++kx) {
            for (int ky = 0; ky < kernel_h; ++ky) {
                int ox = out_x - kx;
                int oy = out_y - ky;
                if (ox % stride_w == 0 && oy % stride_h == 0) {
                    int in_x = ox / stride_w;
                    int in_y = oy / stride_h;
                    if (in_x >= 0 && in_x < input_w && in_y >= 0 && in_y < input_h) {
                        float kernel_value = __ldg(&kernel[ky * kernel_w + kx]);
                        float input_value = __ldg(&input[in_y * input_w + in_x]);
                        sum += input_value * kernel_value;
                    }
                }
            }
        }
        output[out_y * output_w + out_x] = sum;
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

    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    int output_h = (x.size(2) - 1) * stride[0] - 2 * padding[0] + dilation[0] * (weight.size(2) - 1) + 1 + output_padding[0];
    int output_w = (x.size(3) - 1) * stride[1] - 2 * padding[1] + dilation[1] * (weight.size(3) - 1) + 1 + output_padding[1];
    auto output = torch::empty({x.size(0), weight.size(1) * groups, output_h, output_w}, options);

    // Get raw pointers to the data
    float *input_ptr = x.data_ptr<float>();
    float *weight_ptr = weight.data_ptr<float>();
    float *output_ptr = output.data_ptr<float>();

    // Define block and grid sizes
    dim3 threads(16, 16);
    dim3 blocks((output_w + threads.x - 1) / threads.x, (output_h + threads.y - 1) / threads.y);

    // Launch the kernel
    conv_transpose2d_kernel<<<blocks, threads>>>(input_ptr, weight_ptr, output_ptr,
                                                 x.size(3), x.size(2),
                                                 weight.size(3), weight.size(2),
                                                 stride[1], stride[0],
                                                 output_w, output_h);

    // Add bias if provided
    if (bias.has_value()) {
        output.add_(bias.value().reshape({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Fast ConvTranspose2D forward (CUDA)");
}