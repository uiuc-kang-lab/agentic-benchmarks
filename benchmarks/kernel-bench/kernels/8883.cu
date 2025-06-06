#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// Device inline helper functions
__device__ __forceinline__ int gcd(int a, int b) {
  while(b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

__device__ __forceinline__ int my_min(int a, int b) {
  return a < b ? a : b;
}

// Combined CUDA kernel for 2D transposed convolution
// Combines aligned memory accesses, multi-element processing per thread,
// and early exit check for out-of-bound thread indices.
__global__ void combined_conv_transpose2d_kernel(
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

  // Each thread processes 4 output elements
  int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int total = batch * out_channels * out_h * out_w;
  if (index >= total) return;

  #pragma unroll
  for (int i = 0; i < 4 && index + i < total; i++) {
    int idx = index + i;
    // Compute output tensor indices
    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    tmp = tmp / out_h;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    // Start with bias loaded using read-only cache
    float out_val = __ldg(&bias[oc]);
    int g = oc / out_channels_per_group;

    int candidate_h = oh + pad_h;
    int candidate_w = ow + pad_w;

    // Determine the starting kernel position (offset) for height
    int offset_kh = -1;
    int mod_h = candidate_h % stride_h;
    #pragma unroll
    for (int k = 0; k < stride_h; k++) {
      if ((k * dilation_h) % stride_h == mod_h) {
        offset_kh = k;
        break;
      }
    }
    int step_kh = stride_h / gcd(stride_h, dilation_h);
    int kh_bound = candidate_h / dilation_h + 1;
    int kh_end = my_min(kernel_h, kh_bound);

    // Determine the starting kernel position (offset) for width
    int offset_kw = -1;
    int mod_w = candidate_w % stride_w;
    #pragma unroll
    for (int k = 0; k < stride_w; k++) {
      if ((k * dilation_w) % stride_w == mod_w) {
        offset_kw = k;
        break;
      }
    }
    int step_kw = stride_w / gcd(stride_w, dilation_w);
    int kw_bound = candidate_w / dilation_w + 1;
    int kw_end = my_min(kernel_w, kw_bound);

    // Loop over valid kernel height positions
    #pragma unroll
    for (int kh = offset_kh; kh >= 0 && kh < kh_end; kh += step_kh) {
      int h_in_candidate = candidate_h - kh * dilation_h;
      int ih = h_in_candidate / stride_h;
      if (ih >= 0 && ih < in_h) {
        // Loop over valid kernel width positions
        #pragma unroll
        for (int kw = offset_kw; kw >= 0 && kw < kw_end; kw += step_kw) {
          int w_in_candidate = candidate_w - kw * dilation_w;
          int iw = w_in_candidate / stride_w;
          if (iw >= 0 && iw < in_w) {
            // Loop over input channels for this group
            #pragma unroll
            for (int c = g * in_channels_per_group; c < (g + 1) * in_channels_per_group; c++) {
              int x_index = ((n * in_channels + c) * in_h + ih) * in_w + iw;
              int weight_index = ((c * out_channels_per_group + (oc - g * out_channels_per_group)) * kernel_h + kh) * kernel_w + kw;
              out_val += __ldg(&x[x_index]) * __ldg(&weight[weight_index]);
            }
          }
        }
      }
    }
    // Write the computed output value
    int out_index = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
    output[out_index] = out_val;
  }
}

// Host function to launch the kernel
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {

  x = x.contiguous();
  weight = weight.contiguous();
  if (bias.has_value() && bias.value().defined())
    bias = bias.value().contiguous();

  const int batch = x.size(0);
  const int in_channels = x.size(1);
  const int in_h = x.size(2);
  const int in_w = x.size(3);

  const int kernel_h = weight.size(2);
  const int kernel_w = weight.size(3);
  const int out_channels_per_group = weight.size(1);
  const int out_channels = out_channels_per_group * groups;

  const int stride_h = stride[0];
  const int stride_w = stride[1];
  const int pad_h = padding[0];
  const int pad_w = padding[1];
  const int dilation_h = dilation[0];
  const int dilation_w = dilation[1];

  const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
  const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

  if (!bias.has_value() || !bias.value().defined()) {
    bias = at::zeros({out_channels}, weight.options());
  }

  auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());

  int in_channels_per_group = in_channels / groups;

  // Setup kernel launch parameters
  int total_elements = batch * out_channels * out_h * out_w;
  int elements_per_thread = 4;
  int total_threads = (total_elements + elements_per_thread - 1) / elements_per_thread;
  const int threads = 256;
  const int blocks = (total_threads + threads - 1) / threads;

  combined_conv_transpose2d_kernel<<<blocks, threads>>>(
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

  // Check for any kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Combined 2D Transposed Convolution (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
