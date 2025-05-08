#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

inline std::vector<int64_t> parseIntArrayRef(const py::object& obj) {
    std::vector<int64_t> result;
    if (py::isinstance<py::int_>(obj)) {
        result.push_back(obj.cast<int64_t>());
    } else if (py::isinstance<py::sequence>(obj)) {
        for (auto item : obj.cast<py::sequence>()) {
            result.push_back(py::cast<int64_t>(item));
        }
    } else {
        throw std::runtime_error("Expected int or sequence of ints");
    }
    return result;
}

__global__ void conv_transpose2d_kernel(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ output, int N, int C, int H, int W, int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h = threadIdx.y;
    int w = threadIdx.x;

    float value = 0.0f;
    for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
            int in_h = h * stride_h - pad_h + r;
            int in_w = w * stride_w - pad_w + s;
            if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                value += x[((n * C + c) * H + in_h) * W + in_w] * weight[(c * R + r) * S + s];
            }
        }
    }

    // Use warp-level primitive to reduce within a warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        output[((n * C + c) * H + h) * W + w] = value;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0),
    int64_t groups = 1
) {
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);

    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto K = weight.size(0);
    auto R = weight.size(2);
    auto S = weight.size(3);

    auto output = torch::zeros({N, C, H, W}, x.options());

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(N, C);

    conv_transpose2d_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, K, R, S,
        stride_vec[0], stride_vec[1],
        padding_vec[0], padding_vec[1]
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}