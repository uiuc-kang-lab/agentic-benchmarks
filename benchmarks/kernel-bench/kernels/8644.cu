#include <torch/extension.h>
#include <vector>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x);

// Optimized CUDA kernel minimizing warp divergence via branchless control flow
__global__ void transposed_conv3d_branchless_kernel(
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

    // Decode flattened index into coordinates: n, oc, d, h, w
    int tmp = idx;
    int w_out = tmp % out_w; tmp /= out_w;
    int h_out = tmp % out_h; tmp /= out_h;
    int d_out = tmp % out_d; tmp /= out_d;
    int oc = tmp % out_channels; tmp /= out_channels;
    int n = tmp;

    // Initialize output with bias if provided
    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    int group = oc / channels_per_group_out;
    int oc_in_group = oc % channels_per_group_out;

    // Compute base coordinates incorporating padding
    int d_base = d_out + p_d;
    int h_base = h_out + p_h;
    int w_base = w_out + p_w;

    // Compute remainders so that we only iterate over kernel indices that yield valid
    // input coordinates according to the stride. This eliminates in-loop modulo checks.
    int r_d = d_base % s_d;  // valid since d_base is non-negative
    int r_h = h_base % s_h;
    int r_w = w_base % s_w;

    // Loop over kernel depth, height, and width with step = stride to avoid branch divergence
    for (int kd = r_d; kd < k_d; kd += s_d) {
      int in_d_idx = (d_base - kd) / s_d;
      // Branchless validity check: returns 1 if valid, 0 otherwise
      unsigned int valid_d = ((unsigned)in_d_idx < (unsigned)in_d);
      for (int kh = r_h; kh < k_h; kh += s_h) {
        int in_h_idx = (h_base - kh) / s_h;
        unsigned int valid_h = ((unsigned)in_h_idx < (unsigned)in_h);
        for (int kw = r_w; kw < k_w; kw += s_w) {
          int in_w_idx = (w_base - kw) / s_w;
          unsigned int valid_w = ((unsigned)in_w_idx < (unsigned)in_w);
          unsigned int valid = valid_d & valid_h & valid_w;  // 1 if all dimensions valid
          float factor = (float) valid; // 0.0f if any index is out-of-bound, 1.0f otherwise

          // Accumulate over input channels for the current group
          for (int ic = 0; ic < channels_per_group_in; ic++) {
            int in_channel = group * channels_per_group_in + ic;
            int input_idx = n * (in_channels * in_d * in_h * in_w) +
                            in_channel * (in_d * in_h * in_w) +
                            in_d_idx * (in_h * in_w) +
                            in_h_idx * in_w + in_w_idx;
            int weight_idx = in_channel * (channels_per_group_out * k_d * k_h * k_w) +
                             oc_in_group * (k_d * k_h * k_w) +
                             kd * (k_h * k_w) +
                             kh * k_w +
                             kw;
            float in_val = input[input_idx];
            float wt = weight[weight_idx];
            sum += factor * in_val * wt;
          }
        }
      }
    }

    output[idx] = sum;
  }
}

// Host function wrapping the CUDA kernel
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

  const int batch = x.size(0);
  const int in_channels = x.size(1);
  const int in_d = x.size(2);
  const int in_h = x.size(3);
  const int in_w = x.size(4);

  // Weight assumed to be of shape: [in_channels, out_channels_per_group, k_d, k_h, k_w]
  const int k_d = weight.size(2);
  const int k_h = weight.size(3);
  const int k_w = weight.size(4);

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

  const int channels_per_group_out = weight.size(1);
  const int out_channels = channels_per_group_out * groups;
  const int channels_per_group_in = in_channels / groups;

  // Allocate output tensor
  auto output = torch::zeros({batch, out_channels, out_d, out_h, out_w}, x.options());

  const float* x_ptr = x.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const float* bias_ptr = bias_opt.has_value() ? (*bias_opt).data_ptr<float>() : nullptr;
  float* out_ptr = output.data_ptr<float>();

  const int total = batch * out_channels * out_d * out_h * out_w;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  transposed_conv3d_branchless_kernel<<<blocks, threads>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Branchless Transposed Conv3D forward (CUDA)");
}
