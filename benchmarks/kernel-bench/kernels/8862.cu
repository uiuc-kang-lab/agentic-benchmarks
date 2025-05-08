#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>
#include <stdio.h>

// Inline device function to compute greatest common divisor
__device__ int gcd(int a, int b) {
  while (b != 0) {
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

// Optimized CUDA kernel for 2D transposed convolution with loop unrolling
// This kernel is identical to previous optimized version with loop unrolling
__global__ void conv_transpose2d_kernel_unrolled(
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

  // Decode flat index into (n, oc, oh, ow) where n is relative to the sub-batch
  int ow = index % out_w;
  int tmp = index / out_w;
  int oh = tmp % out_h;
  tmp = tmp / out_h;
  int oc = tmp % out_channels;
  int n = tmp / out_channels;

  float out_val = bias[oc];

  // Determine which group this output channel belongs to
  int g = oc / out_channels_per_group;

  // Precompute candidate positions (output coordinate plus padding)
  int candidate_h = oh + pad_h;
  int candidate_w = ow + pad_w;

  // For the height dimension, compute the valid starting offset and step
  int offset_kh = -1;
  int mod_h = candidate_h % stride_h;
  for (int k = 0; k < stride_h; k++) {
    if ((k * dilation_h) % stride_h == mod_h) {
      offset_kh = k;
      break;
    }
  }
  int step_kh = stride_h / gcd(stride_h, dilation_h);
  int kh_bound = candidate_h / dilation_h + 1;
  int kh_end = my_min(kernel_h, kh_bound);

  // For the width dimension, compute the valid starting offset and step
  int offset_kw = -1;
  int mod_w = candidate_w % stride_w;
  for (int k = 0; k < stride_w; k++) {
    if ((k * dilation_w) % stride_w == mod_w) {
      offset_kw = k;
      break;
    }
  }
  int step_kw = stride_w / gcd(stride_w, dilation_w);
  int kw_bound = candidate_w / dilation_w + 1;
  int kw_end = my_min(kernel_w, kw_bound);

  // Iterate over the valid kernel positions in the height dimension with unrolling
  #pragma unroll
  for (int kh = offset_kh; kh >= 0 && kh < kh_end; kh += step_kh) {
    int h_in_candidate = candidate_h - kh * dilation_h;
    int ih = h_in_candidate / stride_h;
    if (ih < 0 || ih >= in_h) continue;

    // Iterate over the valid kernel positions in the width dimension with unrolling
    #pragma unroll
    for (int kw = offset_kw; kw >= 0 && kw < kw_end; kw += step_kw) {
      int w_in_candidate = candidate_w - kw * dilation_w;
      int iw = w_in_candidate / stride_w;
      if (iw < 0 || iw >= in_w) continue;

      // Loop over the corresponding input channels within the group
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

  int out_index = n * (out_channels * out_h * out_w) +
                  oc * (out_h * out_w) +
                  oh * out_w + ow;
  output[out_index] = out_val;
}

// Host function that wraps the CUDA kernel using multiple streams to overlap computation and memory transfers
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

  // If bias was not provided, create a zero tensor
  if (!bias.has_value() || !bias.value().defined()) {
    bias = at::zeros({out_channels}, weight.options());
  }

  auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());

  int in_channels_per_group = in_channels / groups;

  // Determine number of streams to use (e.g. 4 or less if batch is small)
  int num_streams = (batch < 4) ? batch : 4;
  int batch_per_stream = (batch + num_streams - 1) / num_streams;

  // Create CUDA streams
  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // Launch kernels for each sub-batch asynchronously using different streams
  for (int i = 0; i < num_streams; i++) {
    int start = i * batch_per_stream;
    if (start >= batch) break;
    int current_batch = ((start + batch_per_stream) <= batch) ? batch_per_stream : (batch - start);

    int total_threads = current_batch * out_channels * out_h * out_w;
    const int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    // Compute pointer offsets for the current sub-batch
    const float* x_ptr = x.data_ptr<float>() + start * in_channels * in_h * in_w;
    float* output_ptr = output.data_ptr<float>() + start * out_channels * out_h * out_w;

    conv_transpose2d_kernel_unrolled<<<blocks, threads, 0, streams[i]>>>(
        x_ptr,
        weight.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        output_ptr,
        current_batch, // sub-batch size
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
  }

  // Synchronize and destroy all streams
  for (int i = 0; i < num_streams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "2D Transposed Convolution with Stream Overlap (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
