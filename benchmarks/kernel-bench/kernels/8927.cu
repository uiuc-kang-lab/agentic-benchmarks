#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

#define WARP_SIZE 32

// Warp-level reduction using __shfl_down_sync
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel: each warp computes one output element, with warp-level reduction.
__global__ void conv_transpose2d_warp_kernel(
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

    // Each warp computes one output element
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    int total_outputs = batch * out_channels * out_h * out_w;
    if (warp_id >= total_outputs) return;

    // Decode warp_id into (n, oc, oh, ow)
    int tmp = warp_id;
    int ow = tmp % out_w;
    tmp /= out_w;
    int oh = tmp % out_h;
    tmp /= out_h;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    // Determine group from output channel
    int g = oc / out_channels_per_group;

    // Total iterations: over the input channels for this group and kernel area
    int kernel_area = kernel_h * kernel_w;
    int total_iters = in_channels_per_group * kernel_area;

    float sum = 0.0f;
    // Distribute the iterations across warp lanes
    for (int i = lane; i < total_iters; i += WARP_SIZE) {
        int c_offset = i / kernel_area;  // offset within the group's input channels
        int rem = i % kernel_area;
        int kh = rem / kernel_w;
        int kw = rem % kernel_w;
        int c = g * in_channels_per_group + c_offset;

        // Compute the candidate input positions
        int h_in_candidate = oh + pad_h - kh * dilation_h;
        int w_in_candidate = ow + pad_w - kw * dilation_w;

        // Check if candidate positions align with strides
        if ((h_in_candidate % stride_h) != 0 || (w_in_candidate % stride_w) != 0) {
            continue;
        }
        int ih = h_in_candidate / stride_h;
        int iw = w_in_candidate / stride_w;
        if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;

        int x_index = n * (in_channels * in_h * in_w) +
                      c * (in_h * in_w) +
                      ih * in_w + iw;

        // Weight layout: [in_channels, out_channels_per_group, kernel_h, kernel_w]
        int weight_index = c * (out_channels_per_group * kernel_area) +
                           (oc - g * out_channels_per_group) * kernel_area +
                           kh * kernel_w + kw;

        sum += x[x_index] * weight[weight_index];
    }

    // Perform warp-level reduction
    float warp_sum = warpReduceSum(sum);
    // Lane 0 writes the final result
    if (lane == 0) {
        int out_index = n * (out_channels * out_h * out_w) +
                        oc * (out_h * out_w) +
                        oh * out_w + ow;
        output[out_index] = bias[oc] + warp_sum;
    }
}

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

  const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
  const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

  if (!bias.has_value() || !bias.value().defined()) {
    bias = at::zeros({out_channels}, weight.options());
  }

  auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());

  int in_channels_per_group = in_channels / groups;
  int total_outputs = batch * out_channels * out_h * out_w;
  // Launch one warp per output element: 32 threads per warp
  const int threads_per_block = 256;
  int total_threads = total_outputs * WARP_SIZE;
  int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  conv_transpose2d_warp_kernel<<<blocks, threads_per_block>>>(
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
  m.def("forward", &forward, "Warp-level Optimized 2D Transposed Convolution (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
