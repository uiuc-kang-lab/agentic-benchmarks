#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// Inline device function to compute greatest common divisor
__device__ int gcd(int a, int b) {
  while(b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Inline device function for minimum
__device__ int my_min(int a, int b) {
  return a < b ? a : b;
}

// Device function to compute valid kernel offsets and steps
__device__ void compute_offsets_and_steps(int candidate, int stride, int dilation, int kernel_size, int &offset, int &step, int &end) {
  offset = -1;
  int mod = candidate % stride;
  for (int k = 0; k < stride; k++) {
    if ((k * dilation) % stride == mod) {
      offset = k;
      break;
    }
  }
  step = stride / gcd(stride, dilation);
  int bound = candidate / dilation + 1;
  end = my_min(kernel_size, bound);
}

// Device function to perform the convolution operation
__device__ float perform_convolution(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int n, int oc, int oh, int ow,
    int in_channels, int in_h, int in_w,
    int out_channels_per_group, int in_channels_per_group,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int g) {

  float out_val = 0.0f;
  int candidate_h = oh + pad_h;
  int candidate_w = ow + pad_w;

  int offset_kh, step_kh, kh_end;
  compute_offsets_and_steps(candidate_h, stride_h, dilation_h, kernel_h, offset_kh, step_kh, kh_end);

  int offset_kw, step_kw, kw_end;
  compute_offsets_and_steps(candidate_w, stride_w, dilation_w, kernel_w, offset_kw, step_kw, kw_end);

  #pragma unroll
  for (int kh = offset_kh; kh >= 0 && kh < kh_end; kh += step_kh) {
    int h_in_candidate = candidate_h - kh * dilation_h;
    int ih = h_in_candidate / stride_h;
    if (ih < 0 || ih >= in_h) continue;

    #pragma unroll
    for (int kw = offset_kw; kw >= 0 && kw < kw_end; kw += step_kw) {
      int w_in_candidate = candidate_w - kw * dilation_w;
      int iw = w_in_candidate / stride_w;
      if (iw < 0 || iw >= in_w) continue;

      #pragma unroll
      for (int c = g * in_channels_per_group; c < (g + 1) * in_channels_per_group; c++) {
        int x_index = n * (in_channels * in_h * in_w) +
                      c * (in_h * in_w) +
                      ih * in_w + iw;

        int weight_index = c * (out_channels_per_group * kernel_h * kernel_w) +
                           (oc - g * out_channels_per_group) * (kernel_h * kernel_w) +
                           kh * kernel_w + kw;

        out_val += x[x_index] * weight[weight_index];
      }
    }
  }
  return out_val;
}

// Optimized CUDA kernel for 2D transposed convolution with modular device functions
__global__ void conv_transpose2d_kernel_modular(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * out_channels * out_h * out_w;
  if (index >= total) return;

  int ow = index % out_w;
  int tmp = index / out_w;
  int oh = tmp % out_h;
  tmp = tmp / out_h;
  int oc = tmp % out_channels;
  int n = tmp / out_channels;

  float out_val = bias[oc];
  int g = oc / out_channels_per_group;

  out_val += perform_convolution(
      x, weight, n, oc, oh, ow,
      in_channels, in_h, in_w,
      out_channels_per_group, in_channels_per_group,
      kernel_h, kernel_w,
      stride_h, stride_w,
      pad_h, pad_w,
      dilation_h, dilation_w,
      g);

  int out_index = n * (out_channels * out_h * out_w) +
                  oc * (out_h * out_w) +
                  oh * out_w + ow;
  output[out_index] = out_val;
}

// Host function that wraps the CUDA kernel
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {
  
  // Ensure inputs are contiguous
  x = x.contiguous();
  weight = weight.contiguous();
  if (bias.has_value() && bias.value().defined())
    bias = bias.value().contiguous();
  
  // Get dimensions from input and weight
  const int batch = x.size(0);
  const int in_channels = x.size(1);
  const int in_h = x.size(2);
  const int in_w = x.size(3);

  const int kernel_h = weight.size(2);
  const int kernel_w = weight.size(3);
  const int out_channels_per_group = weight.size(1);
  const int out_channels = out_channels_per_group * groups;

  // Retrieve convolution parameters
  const int stride_h = stride[0];
  const int stride_w = stride[1];
  const int pad_h = padding[0];
  const int pad_w = padding[1];
  const int dilation_h = dilation[0];
  const int dilation_w = dilation[1];

  // Compute output dimensions for conv_transpose2d
  const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
  const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

  // If bias is not provided, create a zero tensor
  if (!bias.has_value() || !bias.value().defined()) {
    bias = at::zeros({out_channels}, weight.options());
  }

  auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());

  int in_channels_per_group = in_channels / groups;

  int total_threads = batch * out_channels * out_h * out_w;
  const int threads = 128;
  const int blocks = (total_threads + threads - 1) / threads;

  conv_transpose2d_kernel_modular<<<blocks, threads>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.value().data_ptr<float>(),
      output.data_ptr<float>(),
      batch,
      in_channels,
      in_h,
      in_w,
      out_channels,
      out_h,
      out_w,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      groups,
      in_channels_per_group,
      out_channels_per_group
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized 2D Transposed Convolution with Modular Device Functions (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
