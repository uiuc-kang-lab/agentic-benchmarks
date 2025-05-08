#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Macros for checking tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);   \
  CHECK_CONTIGUOUS(x);

// Inline device function to compute a branchless validity mask
// It computes: index = (coord / stride) if (coord % stride == 0) and index in [0, limit), else returns 0 mask.
// The mask is 1.0f if valid, else 0.0f.
__forceinline__ __device__ float compute_mask(int coord, int stride, int limit, int &index) {
    int div = coord / stride;
    int mod = coord - div * stride;
    // Using ternary operators, which on modern GPUs typically compile to predicated instructions
    float valid_mod = (mod == 0) ? 1.0f : 0.0f;
    float valid_bound = ((div >= 0) && (div < limit)) ? 1.0f : 0.0f;
    index = div;
    return valid_mod * valid_bound;
}

// CUDA kernel implementing transposed conv3d with branchless control flow
__global__ void transposed_conv3d_branchless_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, // may be nullptr
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
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < total; idx += blockDim.x * gridDim.x) {
    // Decode flat index into (n, oc, d, h, w) coordinates
    int tmp = idx;
    int w_out = tmp % out_w; tmp /= out_w;
    int h_out = tmp % out_h; tmp /= out_h;
    int d_out = tmp % out_d; tmp /= out_d;
    int oc = tmp % out_channels; tmp /= out_channels;
    int n = tmp;

    // Initialize accumulation, add bias if available
    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Determine group and intra-group channel index
    int group = oc / channels_per_group_out;
    int oc_in_group = oc % channels_per_group_out;

    // Precompute base coordinates (with padding) in the output space
    int d_base = d_out + p_d;
    int h_base = h_out + p_h;
    int w_base = w_out + p_w;

    // Loop over kernel dimensions without divergent branch: use masks to conditionally add contributions
    for (int kd = 0; kd < k_d; kd++) {
      int in_d_idx;
      float mask_d = compute_mask(d_base - kd, s_d, in_d, in_d_idx);
      for (int kh = 0; kh < k_h; kh++) {
        int in_h_idx;
        float mask_h = compute_mask(h_base - kh, s_h, in_h, in_h_idx);
        for (int kw = 0; kw < k_w; kw++) {
          int in_w_idx;
          float mask_w = compute_mask(w_base - kw, s_w, in_w, in_w_idx);
          float mask = mask_d * mask_h * mask_w;
          
          // For current kernel offset, accumulate contributions from all input channels in this group
          for (int ic = 0; ic < channels_per_group_in; ic++) {
            int in_channel = group * channels_per_group_in + ic;
            // Compute flattened input index: [n, in_channel, in_d_idx, in_h_idx, in_w_idx]
            int input_idx = n * (in_channels * in_d * in_h * in_w) +
                            in_channel * (in_d * in_h * in_w) +
                            in_d_idx * (in_h * in_w) +
                            in_h_idx * in_w +
                            in_w_idx;
            
            // Compute flattened weight index: [in_channel, oc_in_group, kd, kh, kw]
            int weight_idx = in_channel * (channels_per_group_out * k_d * k_h * k_w) +
                             oc_in_group * (k_d * k_h * k_w) +
                             kd * (k_h * k_w) +
                             kh * k_w +
                             kw;
            
            float in_val = input[input_idx];
            float wt = weight[weight_idx];
            sum += mask * in_val * wt;
          }
        }
      }
    }
    
    output[idx] = sum;
  }
}

// C++ forward function wrapping the custom CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,  // not used in kernel computation
    int64_t groups) {

  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  if (bias_opt.has_value()) {
    CHECK_INPUT(*bias_opt);
  }

  // Extract input dimensions
  const int batch = x.size(0);
  const int in_channels = x.size(1);
  const int in_d = x.size(2);
  const int in_h = x.size(3);
  const int in_w = x.size(4);

  // Extract kernel dimensions (assumed weight shape: [in_channels, out_channels_per_group, k_d, k_h, k_w])
  const int k_d = weight.size(2);
  const int k_h = weight.size(3);
  const int k_w = weight.size(4);

  // Stride and padding values
  const int s_d = stride[0];
  const int s_h = stride[1];
  const int s_w = stride[2];
  const int p_d = padding[0];
  const int p_h = padding[1];
  const int p_w = padding[2];
  const int op_d = output_padding[0];
  const int op_h = output_padding[1];
  const int op_w = output_padding[2];

  // Compute output dimensions according to transposed convolution formula
  const int out_d = (in_d - 1) * s_d - 2 * p_d + k_d + op_d;
  const int out_h = (in_h - 1) * s_h - 2 * p_h + k_h + op_h;
  const int out_w = (in_w - 1) * s_w - 2 * p_w + k_w + op_w;

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
  const int total = batch * out_channels * out_d * out_h * out_w;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  // Launch the kernel
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

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward (CUDA, branchless uniform control flow)");
}
