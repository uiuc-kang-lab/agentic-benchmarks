#include <torch/extension.h>
#include <vector>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);     \
  CHECK_CONTIGUOUS(x);

// Optimized CUDA kernel for transposed 3D convolution
// This kernel uses a grid-stride loop with improved control flow by precomputing offsets
// and skipping iterations early using continue statements.
__global__ void optimized_transposed_conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
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

  // Total number of output elements
  int total = batch * out_channels * out_d * out_h * out_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Precompute the spatial sizes
  int out_spatial = out_d * out_h * out_w;
  int in_spatial = in_d * in_h * in_w;
  int input_channel_stride = in_spatial; // For each channel, contiguous block of in_spatial elements
  int output_channel_stride = out_spatial;

  for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
    // Decode the flat index into [n, oc, d, h, w] coordinates
    int tmp = idx;
    int w_idx = tmp % out_w;
    tmp /= out_w;
    int h_idx = tmp % out_h;
    tmp /= out_h;
    int d_idx = tmp % out_d;
    tmp /= out_d;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    // Initialize accumulation with bias if available
    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Determine group and intra-group output channel index
    int group = oc / channels_per_group_out;
    int oc_in_group = oc % channels_per_group_out;

    // Precompute offset values to avoid redundant additions
    int d_offset = d_idx + p_d;
    int h_offset = h_idx + p_h;
    int w_offset = w_idx + p_w;

    // Loop over kernel dimensions with early exit if the corresponding input index is invalid
    for (int kd = 0; kd < k_d; ++kd) {
      int in_d_val = d_offset - kd;
      if (in_d_val % s_d != 0) continue;
      int in_d_idx = in_d_val / s_d;
      if (in_d_idx < 0 || in_d_idx >= in_d) continue;

      for (int kh = 0; kh < k_h; ++kh) {
        int in_h_val = h_offset - kh;
        if (in_h_val % s_h != 0) continue;
        int in_h_idx = in_h_val / s_h;
        if (in_h_idx < 0 || in_h_idx >= in_h) continue;

        for (int kw = 0; kw < k_w; ++kw) {
          int in_w_val = w_offset - kw;
          if (in_w_val % s_w != 0) continue;
          int in_w_idx = in_w_val / s_w;
          if (in_w_idx < 0 || in_w_idx >= in_w) continue;

          // For each valid kernel offset, accumulate contributions over the input channels in the group
          for (int ic = 0; ic < channels_per_group_in; ++ic) {
            int in_channel = group * channels_per_group_in + ic;
            // Compute flattened input index: [n, in_channel, in_d_idx, in_h_idx, in_w_idx]
            int input_index = n * (in_channels * in_spatial) 
                              + in_channel * in_spatial 
                              + in_d_idx * (in_h * in_w) 
                              + in_h_idx * in_w 
                              + in_w_idx;
            float in_val = input[input_index];

            // Compute flattened weight index: [in_channel, oc_in_group, kd, kh, kw]
            int weight_index = in_channel * (channels_per_group_out * k_d * k_h * k_w) 
                               + oc_in_group * (k_d * k_h * k_w) 
                               + kd * (k_h * k_w) 
                               + kh * k_w 
                               + kw;
            float wt = weight[weight_index];
            sum += in_val * wt;
          } // end for ic
        } // end for kw
      } // end for kh
    } // end for kd

    output[idx] = sum;
  } // end for grid-stride loop
}

// C++ forward function wrapping the optimized CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,  // not used in kernel; used for output size calculation
    int64_t groups) {

  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  if (bias_opt.has_value()) {
    CHECK_INPUT(*bias_opt);
  }

  // Get input dimensions
  const int batch = x.size(0);
  const int in_channels = x.size(1);
  const int in_d = x.size(2);
  const int in_h = x.size(3);
  const int in_w = x.size(4);

  // Get kernel dimensions
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

  // Compute output dimensions using convolution transpose formula
  const int out_d = (in_d - 1) * s_d - 2 * p_d + k_d + output_padding[0];
  const int out_h = (in_h - 1) * s_h - 2 * p_h + k_h + output_padding[1];
  const int out_w = (in_w - 1) * s_w - 2 * p_w + k_w + output_padding[2];

  // Determine output channels
  const int channels_per_group_out = weight.size(1);
  const int out_channels = channels_per_group_out * groups;
  const int channels_per_group_in = in_channels / groups;

  // Allocate output tensor
  auto output = torch::zeros({batch, out_channels, out_d, out_h, out_w}, x.options());

  // Get raw pointers
  const float* x_ptr = x.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const float* bias_ptr = bias_opt.has_value() ? (*bias_opt).data_ptr<float>() : nullptr;
  float* out_ptr = output.data_ptr<float>();

  // Total number of output elements
  int total = batch * out_channels * out_d * out_h * out_w;
  
  // Configure CUDA kernel launch parameters
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  optimized_transposed_conv3d_kernel<<<blocks, threads>>>(
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

  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized Transposed Conv3D forward (CUDA)");
}
