/*
Optimized 2D Transposed Convolution combining stream-based partitioning with grid-stride loop for better occupancy.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>
#include <stdio.h>

// Optimized CUDA kernel for 2D transposed convolution
// Uses a grid-stride loop to allow each thread to process multiple output elements
__global__ void conv_transpose2d_optimized_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int total,           // total number of elements (for reference, not used directly)
    const int start,           // starting index for this partition
    const int partition_count, // number of elements in this partition
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

  // Compute the global thread index offset by 'start', then use grid-stride loop
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
       idx < start + partition_count;
       idx += gridDim.x * blockDim.x) {

    // Decode flat index into (n, oc, oh, ow)
    int ow = idx % out_w;
    int temp = idx / out_w;
    int oh = temp % out_h;
    temp = temp / out_h;
    int oc = temp % out_channels;
    int n = temp / out_channels;

    int g = oc / out_channels_per_group;
    float out_val = bias[oc];  // initialize with bias

    // Loop over the input channels belonging to this group
    for (int c = g * in_channels_per_group; c < (g + 1) * in_channels_per_group; c++) {
      for (int kh = 0; kh < kernel_h; kh++) {
        int h_in_candidate = oh + pad_h - kh * dilation_h;
        if (h_in_candidate < 0 || (h_in_candidate % stride_h) != 0) continue;
        int ih = h_in_candidate / stride_h;
        if (ih >= in_h) continue;
        
        for (int kw = 0; kw < kernel_w; kw++) {
          int w_in_candidate = ow + pad_w - kw * dilation_w;
          if (w_in_candidate < 0 || (w_in_candidate % stride_w) != 0) continue;
          int iw = w_in_candidate / stride_w;
          if (iw >= in_w) continue;
          
          int x_index = n * (in_channels * in_h * in_w) +
                        c * (in_h * in_w) +
                        ih * in_w + iw;

          // Weight layout: [in_channels, out_channels_per_group, kernel_h, kernel_w]
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
}

// Forward function to launch the optimized kernel using overlapping CUDA streams
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {
  
  // Ensure tensors are contiguous
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

  // Compute output dimensions
  const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
  const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

  if (!bias.has_value() || !bias.value().defined()) {
    bias = at::zeros({out_channels}, weight.options());
  }

  auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());
  int in_channels_per_group = in_channels / groups;

  int total_elements = batch * out_channels * out_h * out_w;

  // Use multiple streams to overlap computation and memory operations
  int nStreams = 2;  // Number of streams can be tuned based on hardware
  int partition_size = (total_elements + nStreams - 1) / nStreams;
  const int threads = 256;

  std::vector<cudaStream_t> streams(nStreams);
  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // Launch the optimized kernel on each stream with partitioned work
  for (int i = 0; i < nStreams; i++) {
    int start = i * partition_size;
    int count = partition_size;
    if (start + count > total_elements)
      count = total_elements - start;
    if (count <= 0) continue;
    int blocks = (count + threads - 1) / threads;

    conv_transpose2d_optimized_kernel<<<blocks, threads, 0, streams[i]>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        start,
        count,
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
        out_channels_per_group);
  }

  // Synchronize and destroy streams
  for (int i = 0; i < nStreams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized 2D Transposed Convolution (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
