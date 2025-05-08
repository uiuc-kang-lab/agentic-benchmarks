#include <torch/extension.h>
#include <vector>

// Forward function implementing conv_transpose3d with memory coalescing
__global__ void convTranspose3DCoalescedKernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int x_dim0, int x_dim1, int x_dim2, int x_dim3, int x_dim4,
    int w_dim0, int w_dim1, int w_dim2, int w_dim3, int w_dim4,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int out_dim0, int out_dim1, int out_dim2, int out_dim3, int out_dim4
) {
    int batch = blockIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.z * blockDim.z + threadIdx.z;
    int x = threadIdx.x;

    if (z < out_dim2 && y < out_dim3 && x < out_dim4) {
        float value = 0.0f;
        for (int c = 0; c < x_dim1; ++c) {
            for (int kz = 0; kz < w_dim2; ++kz) {
                for (int ky = 0; ky < w_dim3; ++ky) {
                    for (int kx = 0; kx < w_dim4; ++kx) {
                        int in_z = z * stride_d - pad_d + kz;
                        int in_y = y * stride_h - pad_h + ky;
                        int in_x = x * stride_w - pad_w + kx;

                        if (in_z >= 0 && in_z < x_dim2 && in_y >= 0 && in_y < x_dim3 && in_x >= 0 && in_x < x_dim4) {
                            int input_idx = ((batch * x_dim1 + c) * x_dim2 + in_z) * x_dim3 * x_dim4 + in_y * x_dim4 + in_x;
                            int weight_idx = ((c * w_dim1 + blockIdx.y) * w_dim2 + kz) * w_dim3 * w_dim4 + ky * w_dim4 + kx;
                            value += x[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        if (bias != nullptr) {
            value += bias[blockIdx.y];
        }
        int output_idx = ((batch * out_dim1 + blockIdx.y) * out_dim2 + z) * out_dim3 * out_dim4 + y * out_dim4 + x;
        output[output_idx] = value;
    }
}

// Wrapper function for launching the kernel
void convTranspose3DCoalesced(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    torch::Tensor output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding
) {
    const auto batch_size = x.size(0);
    const auto out_channels = weight.size(1);
    const auto out_dim2 = output.size(2);
    const auto out_dim3 = output.size(3);
    const auto out_dim4 = output.size(4);

    dim3 blockSize(32, 8, 8);
    dim3 gridSize(batch_size, out_channels, (out_dim2 + blockSize.y - 1) / blockSize.y);

    convTranspose3DCoalescedKernel<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        x.size(0), x.size(1), x.size(2), x.size(3), x.size(4),
        weight.size(0), weight.size(1), weight.size(2), weight.size(3), weight.size(4),
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        output.size(0), output.size(1), output.size(2), output.size(3), output.size(4)
    );
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convTranspose3DCoalesced", &convTranspose3DCoalesced, "ConvTranspose3D with memory coalescing",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("output"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"));
}