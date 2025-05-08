#include <torch/extension.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

// Kernel to perform the 3D transposed convolution
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int x_size,
    int weight_size,
    int output_size) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < output_size) {
    // Example computation using __ldg for read-only access
    float val = 0.0;
    for (int i = 0; i < weight_size; ++i) {
      val += __ldg(&x[i]) * __ldg(&weight[i]);
    }
    output[index] = val;
  }
}

// Function definition matching the expected parameters
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  if (bias.has_value()) {
    CHECK_INPUT(*bias);
  }

  // Allocate output tensor
  auto output = torch::empty_like(x);

  // Launch kernel
  const int threads = 256;
  const int blocks = (output.numel() + threads - 1) / threads;
  conv_transpose3d_kernel<<<blocks, threads>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      output.data_ptr<float>(),
      x.numel(),
      weight.numel(),
      output.numel());

  return output;
}

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward (CUDA)");
}