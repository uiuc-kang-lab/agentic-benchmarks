#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

// Kernel function for element-wise operation with warp-level reduction
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int input_size,
    int weight_size,
    int output_size,
    int stride,
    int padding,
    int output_padding,
    int groups) {

  extern __shared__ float shared_memory[];
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // Load weights into shared memory
  for (int i = tid; i < weight_size; i += blockDim.x) {
    shared_memory[i] = weight[i];
  }
  __syncthreads();

  // Each block processes one output element
  float result = 0.0f;

  // Iterate over the input and compute the contribution to the output
  for (int i = 0; i < input_size; ++i) {
    result += input[i] * shared_memory[i % weight_size];
  }

  // Warp-level reduction using __shfl_down_sync()
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    result += __shfl_down_sync(0xffffffff, result, offset);
  }

  // Apply bias
  if (tid == 0 && bias != nullptr) {
    result += bias[bid];
  }

  // Store result back to global memory
  if (tid == 0) {
    output[bid] = result;
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

  // Prepare output tensor
  auto output = torch::empty_like(x);

  // Launch kernel
  int threads = 256;
  int blocks = (output.numel() + threads - 1) / threads;
  int shared_mem_size = weight.numel() * sizeof(float);

  conv_transpose3d_kernel<<<blocks, threads, shared_mem_size>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.has_value() ? bias->data_ptr<float>() : nullptr,
      output.data_ptr<float>(),
      x.numel(),
      weight.numel(),
      output.numel(),
      stride[0],
      padding[0],
      output_padding[0],
      groups);

  return output;
}

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward (CUDA)");
}