#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__global__ void conv_transpose2d_kernel(float* x, float* weight, float* output,
                                         int x_height, int x_width, int weight_height, int weight_width,
                                         int output_height, int output_width, int stride_h, int stride_w,
                                         int pad_h, int pad_w) {
    int o_x = blockIdx.x * blockDim.x + threadIdx.x;
    int o_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (o_x < output_height && o_y < output_width) {
        float sum = 0.0f;
        for (int k_h = 0; k_h < weight_height; ++k_h) {
            for (int k_w = 0; k_w < weight_width; ++k_w) {
                int i_x = o_x * stride_h - pad_h + k_h;
                int i_y = o_y * stride_w - pad_w + k_w;
                if (i_x >= 0 && i_x < x_height && i_y >= 0 && i_y < x_width) {
                    sum += x[i_x * x_width + i_y] * weight[k_h * weight_width + k_w];
                }
            }
        }
        output[o_x * output_width + o_y] = sum;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    int x_height = x.size(0);
    int x_width = x.size(1);
    int weight_height = weight.size(0);
    int weight_width = weight.size(1);
    int output_height = (x_height - 1) * stride[0] - 2 * padding[0] + weight_height;
    int output_width = (x_width - 1) * stride[1] - 2 * padding[1] + weight_width;

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor output = torch::zeros({output_height, output_width}, options);

    float* x_data = x.data_ptr<float>();
    float* weight_data = weight.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    dim3 blockSize(16, 16);
    dim3 gridSize((output_height + blockSize.x - 1) / blockSize.x,
                  (output_width + blockSize.y - 1) / blockSize.y);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cudaMemcpyAsync(x_data, x.cpu().data_ptr<float>(), x.numel()*sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(weight_data, weight.cpu().data_ptr<float>(), weight.numel()*sizeof(float), cudaMemcpyHostToDevice, stream1);

    conv_transpose2d_kernel<<<gridSize, blockSize, 0, stream1>>>(x_data, weight_data, output_data,
                                                                 x_height, x_width, weight_height, weight_width,
                                                                 output_height, output_width, stride[0], stride[1],
                                                                 padding[0], padding[1]);

    cudaMemcpyAsync(output.cpu().data_ptr<float>(), output_data, output.numel()*sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);
    cudaStreamDestroy(stream1);

    c10::optional<torch::Tensor> bias = c10::nullopt;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        output += bias.value();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with streams",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}