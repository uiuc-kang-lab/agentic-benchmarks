#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// CUDA kernel that leverages shared memory for weights and bias
__global__ void conv_transpose2d_kernel_shared(
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

  // Each block processes outputs for one group and one batch.
  // Compute the number of output elements for each group.
  int out_per_group = out_channels_per_group * out_h * out_w;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= out_per_group) return;

  // Decode the local output index into (oc_local, oh, ow) within this group
  int oc_local = idx / (out_h * out_w);
  int rem = idx % (out_h * out_w);
  int oh = rem / out_w;
  int ow = rem % out_w;

  int g = blockIdx.y; // group index
  int n = blockIdx.z; // batch index

  // Allocate shared memory: first for weights, then for bias
  // Weight size per group: (in_channels_per_group * out_channels_per_group * kernel_h * kernel_w)
  int weight_size = in_channels_per_group * out_channels_per_group * kernel_h * kernel_w;
  extern __shared__ float s_mem[];
  float* s_weight = s_mem;            // shared weight for this group
  float* s_bias   = s_mem + weight_size; // shared bias for this group

  // Cooperative loading of weight data into shared memory
  for (int j = threadIdx.x; j < weight_size; j += blockDim.x) {
    // Global weight index: each group occupies a contiguous block in the weight tensor
    int global_weight_index = g * in_channels_per_group * (out_channels_per_group * kernel_h * kernel_w) + j;
    s_weight[j] = weight[global_weight_index];
  }

  // Cooperative loading of bias into shared memory
  for (int j = threadIdx.x; j < out_channels_per_group; j += blockDim.x) {
    int global_bias_index = g * out_channels_per_group + j;
    s_bias[j] = bias[global_bias_index];
  }
  __syncthreads();

  // Initialize the accumulator with the bias for this output channel
  float out_val = s_bias[oc_local];
  int h_origin = oh + pad_h;
  int w_origin = ow + pad_w;

  // Loop over the input channels in this group
  for (int ic = 0; ic < in_channels_per_group; ic++) {
    // Loop over kernel height
    for (int kh = 0; kh < kernel_h; kh++) {
      int h_in_candidate = oh + pad_h - kh * dilation_h;
      if (h_in_candidate < 0 || (h_in_candidate % stride_h) != 0) continue;
      int ih = h_in_candidate / stride_h;
      if (ih >= in_h) continue;

      // Loop over kernel width
      for (int kw = 0; kw < kernel_w; kw++) {
        int w_in_candidate = ow + pad_w - kw * dilation_w;
        if (w_in_candidate < 0 || (w_in_candidate % stride_w) != 0) continue;
        int iw = w_in_candidate / stride_w;
        if (iw >= in_w) continue;

        int x_index = n * (in_channels * in_h * in_w) +
                      (g * in_channels_per_group + ic) * (in_h * in_w) +
                      ih * in_w + iw;

        // Compute index into shared weight: weights are stored with layout
        // [in_channels_per_group, out_channels_per_group, kernel_h, kernel_w]
        int weight_index = ic * (out_channels_per_group * kernel_h * kernel_w) +
                           oc_local * (kernel_h * kernel_w) +
                           kh * kernel_w + kw;

        out_val += x[x_index] * s_weight[weight_index];
      } // kw
    } // kh
  } // ic

  int out_index = n * (out_channels * out_h * out_w) +
                  (g * out_channels_per_group + oc_local) * (out_h * out_w) +
                  oh * out_w + ow;
  output[out_index] = out_val;
}


// Forward function that prepares the kernel launch parameters and invokes the CUDA kernel
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

  // Compute output dimensions for transposed convolution
  const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
  const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

  // If bias is not defined, create a zero bias tensor
  if (!bias.has_value() || !bias.value().defined()) {
    bias = at::zeros({out_channels}, weight.options());
  }

  auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());
  int in_channels_per_group = in_channels / groups;

  // Compute the number of output elements per group
  int out_elements_per_group = out_channels_per_group * out_h * out_w;
  const int threads = 256;
  // Grid dimensions: x -> number of tiles for group, y -> group index, z -> batch index
  dim3 grid((out_elements_per_group + threads - 1) / threads, groups, batch);

  // Shared memory size: space for weights and bias for one group
  int shared_mem_size = (in_channels_per_group * out_channels_per_group * kernel_h * kernel_w +
                           out_channels_per_group) * sizeof(float);

  conv_transpose2d_kernel_shared<<<grid, threads, shared_mem_size>>>(
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
      out_channels_per_group);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "2D Transposed Convolution with Shared Memory (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
