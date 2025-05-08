#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// CUDA kernel using warp-level reduction with __shfl_down_sync
__global__ void conv_transpose1d_warp_kernel(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {
  
  // Each warp computes one output element
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = globalThreadId / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  int total_output = N * C_out * L_out;
  if (warp_id >= total_output) return;

  // Decode warp_id into (n, c_out, l_out)
  int tmp = warp_id;
  int n = tmp / (C_out * L_out);
  tmp = tmp % (C_out * L_out);
  int c_out = tmp / L_out;
  int l_out = tmp % L_out;

  // Perform partial reduction over the (C_in x K_w) iterations
  float sum = 0.0f;
  int total_iter = C_in * K_w;
  for (int iter = lane; iter < total_iter; iter += WARP_SIZE) {
    int c_in = iter / K_w;
    int k = iter % K_w;
    int l_in_nom = l_out + padding - k * dilation;
    if (l_in_nom % stride == 0) {
      int l_in = l_in_nom / stride;
      if (l_in >= 0 && l_in < L_in) {
        float x_val = x[n * C_in * L_in + c_in * L_in + l_in];
        float w_val = weight[c_in * C_out * K_w + c_out * K_w + k];
        sum += x_val * w_val;
      }
    }
  }

  // Warp-level reduction using shfl_down_sync (no shared memory required)
  unsigned int mask = 0xffffffff;
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(mask, sum, offset);
  }
  
  // Lane 0 writes the final result
  if (lane == 0) {
    if (bias != nullptr) {
      sum += bias[c_out];
    }
    y[n * C_out * L_out + c_out * L_out + l_out] = sum;
  }
}

// Host function
torch::Tensor conv_transpose1d_forward(
    py::object x_obj,            // x: torch.Tensor
    py::object weight_obj,       // weight: torch.Tensor or torch.nn::Parameter
    py::object bias_obj = py::none(),  // bias: torch.Tensor or None
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1) {

  // Convert py::object to torch::Tensor & ensure contiguity
  torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
  torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();

  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
  TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

  float* bias_ptr = nullptr;
  if (!bias_obj.is_none()) {
    torch::Tensor bias = bias_obj.cast<torch::Tensor>().contiguous();
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
    bias_ptr = bias.data_ptr<float>();
  }

  // Get dimensions
  int N = x.size(0);
  int C_in = x.size(1);
  int L_in = x.size(2);
  int K_w = weight.size(2);
  int C_out = weight.size(1);

  // Compute output length
  int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

  // Allocate output tensor
  auto y = torch::empty({N, C_out, L_out}, x.options());

  // Each warp computes one output element
  int total_output = N * C_out * L_out;
  int warpsNeeded = total_output;
  int threadsPerBlock = 128; // Must be a multiple of warp size
  int totalThreads = warpsNeeded * WARP_SIZE;
  int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

  conv_transpose1d_warp_kernel<<<blocks, threadsPerBlock>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias_ptr,
      y.data_ptr<float>(),
      N, C_in, C_out, L_in, L_out, K_w,
      stride, padding, dilation);

  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &conv_transpose1d_forward,
      "Conv Transpose1D forward (CUDA) with warp-level reduction",
      py::arg("x"),
      py::arg("weight"),
      py::arg("bias") = py::none(),
      py::arg("stride") = 1,
      py::arg("padding") = 0,
      py::arg("dilation") = 1);
}
