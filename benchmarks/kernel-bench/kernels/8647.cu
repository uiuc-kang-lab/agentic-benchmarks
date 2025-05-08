#include <torch/extension.h>
#include <vector>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor!")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous!")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x);

// Optimized CUDA kernel using 3D grid indexing for better thread mapping
__global__ void transposed_conv3d_3d_indexed_kernel(
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

  // Map thread block indices to output spatial coordinates using 3D grid
  int w_out = blockIdx.x * blockDim.x + threadIdx.x;
  int h_out = blockIdx.y * blockDim.y + threadIdx.y;
  
  // The grid's z-dimension encodes a combined index for (batch, out_channels, d_out)
  int index = blockIdx.z;
  int d_out = index % out_d;
  int temp = index / out_d;
  int oc = temp % out_channels;
  int n = temp / out_channels;

  if (w_out >= out_w || h_out >= out_h)
    return;

  float sum = (bias != nullptr) ? bias[oc] : 0.0f;

  int group = oc / channels_per_group_out;
  int oc_in_group = oc % channels_per_group_out;

  // Compute base positions accounting for padding
  int d_base = d_out + p_d;
  int h_base = h_out + p_h;
  int w_base = w_out + p_w;

  // Loop over kernel depth, height, and width
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

        // Loop over the input channels within the group
        for (int ic = 0; ic < channels_per_group_in; ic++) {
          int in_channel = group * channels_per_group_in + ic;

          int input_idx = n * (in_channels * in_d * in_h * in_w) +
                          in_channel * (in_d * in_h * in_w) +
                          in_d_idx * (in_h * in_w) +
                          in_h_idx * in_w +
                          in_w_idx;
          float in_val = input[input_idx];

          int weight_idx = in_channel * (channels_per_group_out * k_d * k_h * k_w) +
                           oc_in_group * (k_d * k_h * k_w) +
                           kd * (k_h * k_w) +
                           kh * k_w +
                           kw;
          float wt = weight[weight_idx];

          sum += in_val * wt;
        }
      }
    }
  }

  int output_idx = n * (out_channels * out_d * out_h * out_w) +
                   oc * (out_d * out_h * out_w) +
                   d_out * (out_h * out_w) +
                   h_out * out_w +
                   w_out;
  output[output_idx] = sum;
}

// C++ forward function wrapping the custom CUDA kernel
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

  // Extract dimensions from input tensor
  int batch = x.size(0);
  int in_channels = x.size(1);
  int in_d = x.size(2);
  int in_h = x.size(3);
  int in_w = x.size(4);

  // Extract kernel dimensions (assumed shape: [in_channels, out_channels_per_group, k_d, k_h, k_w])
  int k_d = weight.size(2);
  int k_h = weight.size(3);
  int k_w = weight.size(4);

  // Get stride and padding values
  int s_d = stride[0];
  int s_h = stride[1];
  int s_w = stride[2];
  int p_d = padding[0];
  int p_h = padding[1];
  int p_w = padding[2];
  int op_d = output_padding[0];
  int op_h = output_padding[1];
  int op_w = output_padding[2];

  // Compute output dimensions for transposed convolution
  int out_d = (in_d - 1) * s_d - 2 * p_d + k_d + op_d;
  int out_h = (in_h - 1) * s_h - 2 * p_h + k_h + op_h;
  int out_w = (in_w - 1) * s_w - 2 * p_w + k_w + op_w;

  int channels_per_group_out = weight.size(1);
  int out_channels = channels_per_group_out * groups;
  int channels_per_group_in = in_channels / groups;

  auto output = torch::zeros({batch, out_channels, out_d, out_h, out_w}, x.options());

  const float* x_ptr = x.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const float* bias_ptr = bias_opt.has_value() ? (*bias_opt).data_ptr<float>() : nullptr;
  float* out_ptr = output.data_ptr<float>();

  // Configure 3D grid and block dimensions
  // Blocks: x for width, y for height, z for combined (batch * out_channels * out_d)
  dim3 block(16, 16, 1);
  dim3 grid((out_w + block.x - 1) / block.x,
            (out_h + block.y - 1) / block.y,
            batch * out_channels * out_d);

  // Launch the kernel
  transposed_conv3d_3d_indexed_kernel<<<grid, block>>>(
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
  m.def("forward", &forward, "Transposed Conv3D forward with 3D thread indexing (CUDA)");
}
