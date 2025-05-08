#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel that uses stream-based pipelining across the batch dimension
// and minimizes warp divergence via arithmetic masks to replace conditionals.
__global__ void conv_transpose1d_kernel_combined(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation,
    int n_start, int n_end) {  // Process batch indices in [n_start, n_end)

  int chunk_N = n_end - n_start; 
  int total_elements = chunk_N * C_out * L_out;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_elements) return;

  // Compute output indices for the current chunk
  int l_out = index % L_out;
  int c_out = (index / L_out) % C_out;
  int n_index = index / (L_out * C_out);
  int n = n_start + n_index;

  // Initialize output value with bias if provided
  float value = (bias != nullptr) ? bias[c_out] : 0.0f;

  // Loop over input channels
  for (int c_in = 0; c_in < C_in; ++c_in) {
    int x_base = n * C_in * L_in + c_in * L_in;
    int w_base = c_in * C_out * K_w + c_out * K_w;
    
    // Loop over kernel positions
    for (int k_w = 0; k_w < K_w; ++k_w) {
      int l_in_nom = l_out + padding - k_w * dilation;
      // Use arithmetic masks to avoid warp divergence
      int mod_cond = ((l_in_nom % stride) == 0) ? 1 : 0;
      int l_in = l_in_nom / stride;
      int range_cond = (l_in >= 0 && l_in < L_in) ? 1 : 0;
      int valid = mod_cond * range_cond;

      // If valid==0, the multiplication will contribute zero
      int x_index = x_base + (valid * l_in);
      float x_val = x[x_index] * valid;
      float w_val = weight[w_base + k_w];
      value += x_val * w_val;
    }
  }

  y[n * C_out * L_out + c_out * L_out + l_out] = value;
}

// Host function using CUDA streams for pipelining the batch processing
torch::Tensor conv_transpose1d_forward(
    py::object x_obj,
    py::object weight_obj,
    py::object bias_obj = py::none(),
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1) {

  // Obtain contiguous tensor inputs
  torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
  torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
  TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

  const float* bias_ptr = nullptr;
  torch::Tensor bias;
  if (!bias_obj.is_none()) {
    bias = bias_obj.cast<torch::Tensor>().contiguous();
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
    bias_ptr = bias.data_ptr<float>();
  }

  // Get tensor dimensions
  int N = static_cast<int>(x.size(0));
  int C_in = static_cast<int>(x.size(1));
  int L_in = static_cast<int>(x.size(2));
  int C_out = static_cast<int>(weight.size(1));
  int K_w = static_cast<int>(weight.size(2));

  // Compute output length
  int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

  // Allocate output tensor
  auto y = torch::empty({N, C_out, L_out}, x.options());

  // Setup kernel launch parameters and stream-based batch pipelining
  int threads = 256;
  int n_streams = 2; // Using two streams for overlapping computation
  int chunk_size = (N + n_streams - 1) / n_streams;

  cudaStream_t streams[2];
  for (int i = 0; i < n_streams; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  for (int i = 0; i < n_streams; i++) {
    int n_start = i * chunk_size;
    int n_end = n_start + chunk_size;
    if (n_end > N) n_end = N;
    if (n_start >= n_end) break;

    int current_chunk = n_end - n_start;
    int total_elements = current_chunk * C_out * L_out;
    int blocks = (total_elements + threads - 1) / threads;

    conv_transpose1d_kernel_combined<<<blocks, threads, 0, streams[i]>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out, K_w,
        stride, padding, dilation,
        n_start, n_end);
  }

  // Synchronize and clean up streams
  for (int i = 0; i < n_streams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &conv_transpose1d_forward,
      "Combined Conv Transpose1D forward (CUDA) with stream pipelining and minimized warp divergence",
      py::arg("x"),
      py::arg("weight"),
      py::arg("bias") = py::none(),
      py::arg("stride") = 1,
      py::arg("padding") = 0,
      py::arg("dilation") = 1);
}
