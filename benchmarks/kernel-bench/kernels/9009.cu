#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function that performs the convolution for one output element.
__device__ __forceinline__ float compute_conv1d(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int b,
    int oc,
    int o,
    int in_channels,
    int in_size,
    int kernel_size,
    int stride,
    int dilation) {
  float sum = 0.0f;
  int start_pos = o * stride;
  int end_pos = start_pos + (kernel_size - 1) * dilation;

  // If the convolution window is fully in bounds, avoid per-element boundary checks
  if (end_pos < in_size) {
    for (int ic = 0; ic < in_channels; ++ic) {
      const float* x_ptr = x + b * (in_channels * in_size) + ic * in_size + start_pos;
      const float* w_ptr = weight + oc * in_channels * kernel_size + ic * kernel_size;
      #pragma unroll
      for (int k = 0; k < kernel_size; ++k) {
        sum += x_ptr[k * dilation] * w_ptr[k];
      }
    }
  } else {
    // In boundary cases, check each position
    for (int ic = 0; ic < in_channels; ++ic) {
      const float* x_ptr = x + b * (in_channels * in_size) + ic * in_size;
      const float* w_ptr = weight + oc * in_channels * kernel_size + ic * kernel_size;
      #pragma unroll
      for (int k = 0; k < kernel_size; ++k) {
        int pos = start_pos + k * dilation;
        if (pos < in_size) {
          sum += x_ptr[pos] * w_ptr[k];
        }
      }
    }
  }
  return sum;
}

// CUDA kernel that computes the 1D convolution using shared memory for weights
__global__ void conv1d_shared_memory_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation) {
  extern __shared__ float shared_weight[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = B * out_channels * out_size;
  if (idx >= total_elements) return;

  // Decode the linear index into (b, oc, o)
  int o = idx % out_size;
  int tmp = idx / out_size;
  int oc = tmp % out_channels;
  int b = tmp / out_channels;

  // Load weights into shared memory
  for (int ic = threadIdx.x; ic < in_channels * kernel_size; ic += blockDim.x) {
    shared_weight[ic] = weight[oc * in_channels * kernel_size + ic];
  }
  __syncthreads();

  // Compute the convolution using the modular device function
  float sum = compute_conv1d(x, shared_weight, b, oc, o, in_channels, in_size, kernel_size, stride, dilation);

  // Add bias if provided
  if (bias != nullptr) {
    sum += bias[oc];
  }

  int out_idx = b * (out_channels * out_size) + oc * out_size + o;
  output[out_idx] = sum;
}

// Forward function exposed via pybind11
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation) {
  TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(x.dim() == 3, "x must be 3D");
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
  TORCH_CHECK(weight.size(1) == x.size(1), "Input channels mismatch");

  if (bias.has_value()) {
    TORCH_CHECK(bias.value().device().is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
    TORCH_CHECK(bias.value().size(0) == weight.size(0), "Bias size mismatch");
  }

  int B = x.size(0);
  int in_channels = x.size(1);
  int in_size = x.size(2);
  int out_channels = weight.size(0);
  int kernel_size = weight.size(2);

  int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
  TORCH_CHECK(out_size > 0, "Invalid output size");

  auto output = torch::empty({B, out_channels, out_size}, x.options());
  if (output.numel() == 0) return output;

  const float* x_data = x.data_ptr<float>();
  const float* weight_data = weight.data_ptr<float>();
  const float* bias_data = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
  float* output_data = output.data_ptr<float>();

  int total_elements = B * out_channels * out_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  size_t shared_memory_size = in_channels * kernel_size * sizeof(float);

  conv1d_shared_memory_kernel<<<blocks, threads, shared_memory_size>>>(
      x_data,
      weight_data,
      bias_data,
      output_data,
      B,
      in_channels,
      in_size,
      out_channels,
      kernel_size,
      out_size,
      stride,
      dilation);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Modular 1D convolution forward with shared memory (CUDA)");
}
