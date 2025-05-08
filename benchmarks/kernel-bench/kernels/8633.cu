#include <torch/extension.h>
#include <vector>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x);

// Define a compile-time block size that can be tuned (32, 64, 128, 256, 512)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// CUDA kernel using a configurable block size via the BLOCK_SIZE macro
__global__ void blocksize_experiment_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr
    float* __restrict__ output,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int out_d,
    int out_h,
    int out_w,
    int k_d,
    int k_h,
    int k_w,
    int s_d,
    int s_h,
    int s_w,
    int p_d,
    int p_h,
    int p_w,
    int groups,
    int channels_per_group_in,
    int channels_per_group_out) {

  int total = batch * out_channels * out_d * out_h * out_w;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int tmp = idx;
    int w_out = tmp % out_w; tmp /= out_w;
    int h_out = tmp % out_h; tmp /= out_h;
    int d_out = tmp % out_d; tmp /= out_d;
    int oc = tmp % out_channels; tmp /= out_channels;
    int n = tmp;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    int group = oc / channels_per_group_out;
    int oc_in_group = oc % channels_per_group_out;

    int d_base = d_out + p_d;
    int h_base = h_out + p_h;
    int w_base = w_out + p_w;

    for (int kd = 0; kd < k_d; kd++) {
      int tmp_d = d_base - kd;
      if (tmp_d % s_d != 0) continue;
      int in_d_idx = tmp_d / s_d;
      if (in_d_idx < 0 || in_d_idx >= in_d) continue;

      for (int kh = 0; kh < k_h; kh++) {
        int tmp_h = h_base - kh;
        if (tmp_h % s_h != 0) continue;
        int in_h_idx = tmp_h / s_h;
        if (in_h_idx < 0 || in_h_idx >= in_h) continue;

        for (int kw = 0; kw < k_w; kw++) {
          int tmp_w = w_base - kw;
          if (tmp_w % s_w != 0) continue;
          int in_w_idx = tmp_w / s_w;
          if (in_w_idx < 0 || in_w_idx >= in_w) continue;

          for (int ic = 0; ic < channels_per_group_in; ic++) {
            int in_channel = group * channels_per_group_in + ic;

            int input_idx = n * (in_channels * in_d * in_h * in_w) +
                            in_channel * (in_d * in_h * in_w) +
                            in_d_idx * (in_h * in_w) +
                            in_h_idx * in_w + in_w_idx;

            float in_val = input[input_idx];

            int weight_idx = in_channel * (channels_per_group_out * k_d * k_h * k_w) +
                             oc_in_group * (k_d * k_h * k_w) +
                             kd * (k_h * k_w) + kh * k_w + kw;

            float wt = weight[weight_idx];
            sum += in_val * wt;
          }
        }
      }
    }

    output[idx] = sum;
  }
}

// C++ forward function wrapping the CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  if (bias_opt.has_value()) {
    CHECK_INPUT(*bias_opt);
  }

  // Input dimensions (assumed shape: [batch, in_channels, in_d, in_h, in_w])
  const int batch = x.size(0);
  const int in_channels = x.size(1);
  const int in_d = x.size(2);
  const int in_h = x.size(3);
  const int in_w = x.size(4);

  // Kernel dimensions (assumed shape: [in_channels, out_channels_per_group, k_d, k_h, k_w])
  const int k_d = weight.size(2);
  const int k_h = weight.size(3);
  const int k_w = weight.size(4);

  // Stride and padding
  const int s_d = stride[0];
  const int s_h = stride[1];
  const int s_w = stride[2];
  const int p_d = padding[0];
  const int p_h = padding[1];
  const int p_w = padding[2];
  const int op_d = output_padding[0];
  const int op_h = output_padding[1];
  const int op_w = output_padding[2];

  // Compute output dimensions for transposed convolution
  const int out_d = (in_d - 1) * s_d - 2 * p_d + k_d + op_d;
  const int out_h = (in_h - 1) * s_h - 2 * p_h + k_h + op_h;
  const int out_w = (in_w - 1) * s_w - 2 * p_w + k_w + op_w;

  // Determine output channels (weight shape: [in_channels, out_channels_per_group, k_d, k_h, k_w])
  const int channels_per_group_out = weight.size(1);
  const int out_channels = channels_per_group_out * groups;
  const int channels_per_group_in = in_channels / groups;

  // Allocate output tensor
  auto output = torch::zeros({batch, out_channels, out_d, out_h, out_w}, x.options());

  // Get raw pointers for kernel launch
  const float* x_ptr = x.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const float* bias_ptr = (bias_opt.has_value()) ? (*bias_opt).data_ptr<float>() : nullptr;
  float* out_ptr = output.data_ptr<float>();

  const int total = batch * out_channels * out_d * out_h * out_w;
  const int threads = BLOCK_SIZE;
  const int blocks = (total + threads - 1) / threads;

  // Launch kernel with the compile-time defined block size
  blocksize_experiment_kernel<<<blocks, threads>>>(
      x_ptr,
      weight_ptr,
      bias_ptr,
      out_ptr,
      batch,
      in_channels,
      in_d,
      in_h,
      in_w,
      out_channels,
      out_d,
      out_h,
      out_w,
      k_d,
      k_h,
      k_w,
      s_d,
      s_h,
      s_w,
      p_d,
      p_h,
      p_w,
      groups,
      channels_per_group_in,
      channels_per_group_out);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Blocksize Experiment Transposed Conv3D forward (CUDA)");
}
