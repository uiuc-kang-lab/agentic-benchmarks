#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>
#include <cstdio>

// Optimized CUDA kernel using shared memory to load the weight tile for each output channel
// Each block computes a tile of output spatial positions for a given (batch, output channel) pair
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

  // Decode the blockIdx.z to determine which (batch, output channel) pair we are processing
  int combined = blockIdx.z;
  int n = combined / out_channels;
  int oc = combined % out_channels;

  // Compute the starting position of the output tile
  int tile_start_w = blockIdx.x * blockDim.x;
  int tile_start_h = blockIdx.y * blockDim.y;
  int ow = tile_start_w + threadIdx.x;
  int oh = tile_start_h + threadIdx.y;

  // Allocate shared memory to preload the weight tile for the current output channel
  extern __shared__ float s_weight[];
  // Total number of weight elements for this output channel across its corresponding input channels
  int total_weight = in_channels_per_group * kernel_h * kernel_w;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  
  // Each thread loads elements from global memory into shared memory in a strided loop
  for (int i = tid; i < total_weight; i += blockDim.x * blockDim.y) {
    int c_local = i / (kernel_h * kernel_w); // local index within the input channels for the group
    int rem = i % (kernel_h * kernel_w);
    int kh = rem / kernel_w;
    int kw = rem % kernel_w;
    
    // Determine the group index and actual input channel index
    int g = oc / out_channels_per_group;
    int c = g * in_channels_per_group + c_local;
    
    // weight layout: [in_channels, out_channels_per_group, kernel_h, kernel_w]
    // For a given input channel c, the weight for output channel oc is:
    // index = c * (out_channels_per_group * kernel_h * kernel_w) + (oc - g * out_channels_per_group) * (kernel_h * kernel_w) + kh * kernel_w + kw
    int weight_index = c * (out_channels_per_group * kernel_h * kernel_w) +
                       (oc - g * out_channels_per_group) * (kernel_h * kernel_w) +
                       kh * kernel_w + kw;
    s_weight[i] = weight[weight_index];
  }
  __syncthreads();

  // If this thread's output coordinate is outside the valid range, exit
  if (oh >= out_h || ow >= out_w)
    return;

  // Load bias for the current output channel
  float out_val = bias[oc];

  int g = oc / out_channels_per_group;
  // Loop over the input channels of the corresponding group
  for (int c_local = 0; c_local < in_channels_per_group; c_local++) {
    int c = g * in_channels_per_group + c_local;
    // Loop over kernel height
    for (int kh = 0; kh < kernel_h; kh++) {
      int h_in_candidate = oh + pad_h - kh * dilation_h;
      if (h_in_candidate < 0 || (h_in_candidate % stride_h) != 0)
        continue;
      int ih = h_in_candidate / stride_h;
      if (ih >= in_h)
        continue;
      // Loop over kernel width
      for (int kw = 0; kw < kernel_w; kw++) {
        int w_in_candidate = ow + pad_w - kw * dilation_w;
        if (w_in_candidate < 0 || (w_in_candidate % stride_w) != 0)
          continue;
        int iw = w_in_candidate / stride_w;
        if (iw >= in_w)
          continue;

        int x_index = n * (in_channels * in_h * in_w) +
                      c * (in_h * in_w) +
                      ih * in_w + iw;
        // Compute index in shared memory for this weight element
        int s_index = c_local * (kernel_h * kernel_w) + kh * kernel_w + kw;
        out_val += x[x_index] * s_weight[s_index];
      }
    }
  }

  int out_index = n * (out_channels * out_h * out_w) +
                  oc * (out_h * out_w) +
                  oh * out_w + ow;
  output[out_index] = out_val;
}

// Forward function setting up the grid and launching the kernel
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {
  
  // Ensure contiguous tensors
  x = x.contiguous();
  weight = weight.contiguous();
  if (bias.has_value() && bias.value().defined())
    bias = bias.value().contiguous();
  
  // Extract dimensions
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

  // Compute output dimensions
  const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
  const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

  // Create bias tensor if not provided
  if (!bias.has_value() || !bias.value().defined()) {
    bias = at::zeros({out_channels}, weight.options());
  }

  auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());
  int in_channels_per_group = in_channels / groups;
  
  // Define tile dimensions for the output spatial domain
  const int TILE_W = 16;
  const int TILE_H = 16;
  dim3 block(TILE_W, TILE_H);
  dim3 grid((out_w + TILE_W - 1) / TILE_W,
            (out_h + TILE_H - 1) / TILE_H,
            batch * out_channels);
  
  // Allocate shared memory for the weight tile
  size_t shared_mem_size = in_channels_per_group * kernel_h * kernel_w * sizeof(float);

  conv_transpose2d_kernel_shared<<<grid, block, shared_mem_size>>>(
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
    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
  }
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "2D Transposed Convolution with Shared Memory Optimization (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
