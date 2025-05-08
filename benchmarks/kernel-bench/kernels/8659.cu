#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

// CUDA kernel for 3D transposed convolution with improved memory coalescing
__global__ void conv_transpose3d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr if bias not provided
    float* __restrict__ output,
    int N, int C_in,
    int D_in, int H_in, int W_in,
    int C_out,
    int D_out, int H_out, int W_out,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups) {

  // Total number of elements in the output tensor
  int total_elements = N * C_out * D_out * H_out * W_out;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  // Map linear index to multi-dimensional indices (w is the fastest changing dimension)
  int w = idx % W_out;
  int temp = idx / W_out;
  int h = temp % H_out;
  temp = temp / H_out;
  int d = temp % D_out;
  temp = temp / D_out;
  int c = temp % C_out;
  int n = temp / C_out;

  float value = 0.0f;
  if (bias != nullptr) {
      value = bias[c];
  }

  // Determine group parameters
  int group_size_in = C_in / groups;
  int group_size_out = C_out / groups;
  int group = c / group_size_out;
  int c_weight = c - group * group_size_out;  // index within the group

  // Iterate over kernel dimensions, mapping output coordinate to corresponding input positions
  for (int kd = 0; kd < Kd; kd++) {
    int id = d + pad_d - kd;
    if (id % stride_d != 0) continue;
    id /= stride_d;
    if (id < 0 || id >= D_in) continue;

    for (int kh = 0; kh < Kh; kh++) {
      int ih = h + pad_h - kh;
      if (ih % stride_h != 0) continue;
      ih /= stride_h;
      if (ih < 0 || ih >= H_in) continue;

      for (int kw = 0; kw < Kw; kw++) {
        int iw = w + pad_w - kw;
        if (iw % stride_w != 0) continue;
        iw /= stride_w;
        if (iw < 0 || iw >= W_in) continue;

        // For each corresponding input channel in the current group
        for (int ci = 0; ci < group_size_in; ci++) {
          int c_in = group * group_size_in + ci;
          // Compute the flat index for the input tensor (N, C_in, D_in, H_in, W_in)
          int input_idx = (((n * C_in + c_in) * D_in + id) * H_in + ih) * W_in + iw;
          // Compute the flat index for the weight tensor (C_in, C_out_per_group, Kd, Kh, Kw)
          int weight_idx = ((((c_in) * group_size_out + c_weight) * Kd + kd) * Kh + kh) * Kw + kw;
          
          value += input[input_idx] * weight[weight_idx];
        }
      }
    }
  }

  // Write the computed result to the output tensor (N, C_out, D_out, H_out, W_out).
  int output_idx = (((n * C_out + c) * D_out + d) * H_out + h) * W_out + w;
  output[output_idx] = value;
}

// Host function that wraps the CUDA kernel launch
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

  // Input dimensions: x is assumed to be of shape (N, C_in, D_in, H_in, W_in)
  int N = x.size(0);
  int C_in = x.size(1);
  int D_in = x.size(2);
  int H_in = x.size(3);
  int W_in = x.size(4);

  // Weight dimensions: weight is assumed to be of shape (C_in, C_out_per_group, Kd, Kh, Kw)
  int Kd = weight.size(2);
  int Kh = weight.size(3);
  int Kw = weight.size(4);
  int C_out = weight.size(1) * groups; // Total output channels

  // Extract stride, padding, and output padding values
  int stride_d = stride[0];
  int stride_h = stride[1];
  int stride_w = stride[2];

  int pad_d = padding[0];
  int pad_h = padding[1];
  int pad_w = padding[2];

  int out_pad_d = output_padding[0];
  int out_pad_h = output_padding[1];
  int out_pad_w = output_padding[2];

  // Compute output dimensions for conv_transpose3d
  int D_out = (D_in - 1) * stride_d - 2 * pad_d + Kd + out_pad_d;
  int H_out = (H_in - 1) * stride_h - 2 * pad_h + Kh + out_pad_h;
  int W_out = (W_in - 1) * stride_w - 2 * pad_w + Kw + out_pad_w;

  auto options = torch::TensorOptions()
                      .dtype(x.dtype())
                      .device(x.device());
  auto output = torch::empty({N, C_out, D_out, H_out, W_out}, options);

  // Determine total number of output elements and launch parameters
  int total_elements = output.numel();
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  // Retrieve raw pointers
  const float* input_ptr = x.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  float* output_ptr = output.data_ptr<float>();

  // Launch the CUDA kernel with improved memory coalescing by aligning output writes
  conv_transpose3d_forward_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input_ptr, weight_ptr, bias_ptr, output_ptr,
      N, C_in, D_in, H_in, W_in,
      C_out,
      D_out, H_out, W_out,
      Kd, Kh, Kw,
      stride_d, stride_h, stride_w,
      pad_d, pad_h, pad_w,
      out_pad_d, out_pad_h, out_pad_w,
      groups);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward with memory coalescing (CUDA)");
}
