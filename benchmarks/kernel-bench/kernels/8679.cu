#include <torch/extension.h>

// Combined macro to ensure a tensor is both a contiguous CUDA tensor.
#define CHECK_CUDA_INPUT(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be a contiguous CUDA tensor")

// Optimized forward function that minimizes conditional branches and unifies input checks.
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

  // Unified input checks to ensure uniform control flow and reduce warp divergence
  CHECK_CUDA_INPUT(x);
  CHECK_CUDA_INPUT(weight);

  // Evaluate bias uniformly: if provided, check its properties; otherwise, use an empty tensor
  torch::Tensor bias_tensor = at::Tensor();
  if (bias.has_value()) {
    CHECK_CUDA_INPUT(*bias);
    bias_tensor = *bias;
  }

  // Call the underlying transposed 3D convolution function
  return at::conv_transpose3d(
      x,
      weight,
      bias_tensor,
      stride,
      padding,
      output_padding,
      groups
  );
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized Transposed Conv3D forward (CUDA) with unified input checks");
}
