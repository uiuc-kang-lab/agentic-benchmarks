#include <torch/extension.h>
#include <vector>

// Forward function implementing conv_transpose3d
__global__ void convTranspose3DKernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int X, int Y, int Z,
    int FX, int FY, int FZ,
    int stride_x, int stride_y, int stride_z,
    int pad_x, int pad_y, int pad_z,
    int output_x, int output_y, int output_z,
    int batch_size, int num_kernels,
    int kernel_size, int groups
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch_size * num_kernels * output_z * output_y * output_x) {
        int temp = n;
        int px = temp % output_x;
        temp /= output_x;
        int py = temp % output_y;
        temp /= output_y;
        int pz = temp % output_z;
        temp /= output_z;
        int c = temp % num_kernels;
        int b = temp / num_kernels;

        int xb = b * groups;

        float value = 0;
        for (int dz = 0; dz < FZ; ++dz) {
          for (int dy = 0; dy < FY; ++dy) {
            for (int dx = 0; dx < FX; ++dx) {
              int ox = px * stride_x + dx - pad_x;
              int oy = py * stride_y + dy - pad_y;
              int oz = pz * stride_z + dz - pad_z;

              if (ox >= 0 && ox < X && oy >= 0 && oy < Y && oz >= 0 && oz < Z) {
                int input_idx = (((b * (X * Y * Z)) + (oz * Y * X) + (oy * X) + ox) * num_kernels + c);
                int weight_idx = (((c * FZ + dz) * FY + dy) * FX + dx);
                value += x[input_idx] * weight[weight_idx];
              }
            }
          }
        }
        int output_idx = (((b * num_kernels + c) * output_z + pz) * output_y + py) * output_x + px;
        output[output_idx] = value;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    const auto batch_size = x.size(0);
    const auto input_channels = x.size(1);
    const auto depth = x.size(2);
    const auto height = x.size(3);
    const auto width = x.size(4);

    const auto output_channels = weight.size(0);
    const auto FZ = weight.size(2);
    const auto FY = weight.size(3);
    const auto FX = weight.size(4);

    const int OX = (width - 1) * stride[0] - 2 * padding[0] + FX;
    const int OY = (height - 1) * stride[1] - 2 * padding[1] + FY;
    const int OZ = (depth - 1) * stride[2] - 2 * padding[2] + FZ;

    auto output = torch::zeros({batch_size, output_channels, OZ, OY, OX}, torch::Device(torch::kCUDA));

    int blocks = (batch_size * output_channels * OZ * OY * OX + 255) / 256;
    convTranspose3DKernel<<<blocks, 256>>>(
        x.contiguous().data_ptr<float>(),
        weight.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        depth, height, width,
        FX, FY, FZ,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        OX, OY, OZ,
        batch_size, output_channels,
        FX * FY * FZ, groups
    );

    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward function",
          py::arg("x"),
          py::arg("weight"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}