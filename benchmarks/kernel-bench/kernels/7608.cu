#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {
__device__ __forceinline__ int64_t compute_output_index(
    int64_t input_index,
    int64_t stride,
    int64_t padding,
    int64_t kernel_size) {
  return input_index * stride - padding + kernel_size - 1;
}

__device__ __forceinline__ bool is_valid_position(
    int64_t pos,
    int64_t lower_bound,
    int64_t upper_bound) {
  return pos >= lower_bound && pos < upper_bound;
}

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t in_depth,
    int64_t in_height,
    int64_t in_width,
    int64_t out_channels,
    int64_t out_depth,
    int64_t out_height,
    int64_t out_width,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w) {

  const int64_t n = blockIdx.x;
  const int64_t c = blockIdx.y;
  int64_t spatial_index = blockIdx.z * blockDim.x + threadIdx.x;
  if (spatial_index >= out_depth * out_height * out_width) return;
  const int64_t d = spatial_index / (out_height * out_width);
  const int64_t h = (spatial_index % (out_height * out_width)) / out_width;
  const int64_t w = spatial_index % out_width;

  if (n >= batch_size || c >= out_channels || d >= out_depth || h >= out_height || w >= out_width) return;

  float value = 0.0f;

  for (int64_t kd = 0; kd < kernel_d; ++kd) {
    for (int64_t kh = 0; kh < kernel_h; ++kh) {
      for (int64_t kw = 0; kw < kernel_w; ++kw) {
        const int64_t in_d = compute_output_index(d, stride_d, padding_d, kernel_d) - kd;
        const int64_t in_h = compute_output_index(h, stride_h, padding_h, kernel_h) - kh;
        const int64_t in_w = compute_output_index(w, stride_w, padding_w, kernel_w) - kw;

        if (is_valid_position(in_d, 0, in_depth) &&
            is_valid_position(in_h, 0, in_height) &&
            is_valid_position(in_w, 0, in_width)) {
          for (int64_t g = 0; g < in_channels; ++g) {
            const int64_t input_idx = ((n * in_channels + g) * in_depth + in_d) * in_height * in_width
                                    + in_h * in_width + in_w;
            const int64_t weight_idx = ((c * in_channels + g) * kernel_d + kd) * kernel_h * kernel_w
                                    + kh * kernel_w + kw;
            value += input[input_idx] * weight[weight_idx];
          }
        }
      }
    }
  }

  if (bias) {
    value += bias[c];
  }

  const int64_t output_idx = ((n * out_channels + c) * out_depth + d) * out_height * out_width
                           + h * out_width + w;
  output[output_idx] = value;
}
} // namespace

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

  auto input = x.contiguous();
  auto weight_ = weight.contiguous();
  auto output = torch::empty(
      {input.size(0),
       weight.size(1) * groups,
       (input.size(2) - 1) * stride[0] - 2 * padding[0] + output_padding[0] + weight.size(2),
       (input.size(3) - 1) * stride[1] - 2 * padding[1] + output_padding[1] + weight.size(3),
       (input.size(4) - 1) * stride[2] - 2 * padding[2] + output_padding[2] + weight.size(4)},
      input.options());

  const int64_t batch_size = input.size(0);
  const int64_t in_channels = input.size(1);
  const int64_t out_channels = weight.size(1) * groups;
  const auto [in_depth, in_height, in_width] = std::make_tuple(input.size(2), input.size(3), input.size(4));
  const auto [kernel_d, kernel_h, kernel_w] = std::make_tuple(weight.size(2), weight.size(3), weight.size(4));

  const dim3 blocks(
      batch_size,
      (out_channels + 3) / 4,
      (output.size(2) * output.size(3) + 7) / 8,
      (output.size(4) + 15) / 16);
  const dim3 threads(1, 4, 8, 16);

  conv_transpose3d_kernel<<<blocks, threads>>>(
      input.data_ptr<float>(),
      weight_.data_ptr<float>(),
      bias ? bias->data_ptr<float>() : nullptr,
      output.data_ptr<float>(),
      batch_size,
      in_channels,
      in_depth,
      in_height,
      in_width,
      out_channels,
      output.size(2),
      output.size(3),
      output.size(4),
      kernel_d,
      kernel_h,
      kernel_w,
      stride[0],
      stride[1],
      stride[2],
      padding[0],
      padding[1],
      padding[2]);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "ConvTranspose3d forward function",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = nullptr,
        py::arg("stride"),
        py::arg("padding"),
        py::arg("output_padding"),
        py::arg("groups"));
}