#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

__device__ int gcd(int a, int b) {
  while (b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

__device__ int my_min(int a, int b) {
  return a < b ? a : b;
}

// Optimized kernel with atomic reduction in shared memory
__global__ void conv_transpose2d_kernel_atomic(
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

  extern __shared__ float shared_out[];

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * out_channels * out_h * out_w;
  if (index >= total) return;

  int ow = index % out_w;
  int tmp = index / out_w;
  int oh = tmp % out_h;
  tmp = tmp / out_h;
  int oc = tmp % out_channels;
  int n = tmp / out_channels;

  // Use shared memory to store intermediate results to reduce global atomic operations
  int shared_index = threadIdx.x % blockDim.x;
  if (shared_index < blockDim.x) {
    shared_out[shared_index] = 0.0;
  }

  __syncthreads();  // Ensure all loads to shared memory are finished

  int g = oc / out_channels_per_group;
  int candidate_h = oh + pad_h;
  int candidate_w = ow + pad_w;

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

  float out_val = 0.0f;

  for (int c = g * in_channels_per_group; c < (g + 1) * in_channels_per_group; c++) {
    for (int kh = offset_kh; kh >= 0 && kh < kh_end; kh += step_kh) {
      int h_in_candidate = candidate_h - kh * dilation_h;
      if (h_in_candidate < 0 || h_in_candidate % stride_h != 0) continue;
      int ih = h_in_candidate / stride_h;
      if (ih >= in_h) continue;

      for (int kw = offset_kw; kw >= 0 && kw < kw_end; kw += step_kw) {
        int w_in_candidate = candidate_w - kw * dilation_w;
        if (w_in_candidate < 0 || w_in_candidate % stride_w != 0) continue;
        int iw = w_in_candidate / stride_w;
        if (iw >= in_w) continue;

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

  atomicAdd(&shared_out[shared_index], out_val);

  __syncthreads();

  // Final reduction from shared memory to global memory
  if (threadIdx.x % blockDim.x == 0) {
    for (int i = 1; i < blockDim.x; ++i) {
      out_val += shared_out[i];
    }
    int out_index = n * (out_channels * out_h * out_w) +
                    oc * (out_h * out_w) +
                    oh * out_w + ow;
    output[out_index] += out_val;
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

  int total_threads = batch * out_channels * out_h * out_w;
  const int threads = 256;
  const int blocks = (total_threads + threads - 1) / threads;

  size_t shared_memory_size = threads * sizeof(float);

  conv_transpose2d_kernel_atomic<<<blocks, threads, shared_memory_size>>>(
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
  m.def("forward", &forward, "Optimized 2D Transposed Convolution with Atomics (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}