#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Modular device function: computes the output value for one element in the transposed convolution
template <typename scalar_t>
__device__ inline scalar_t compute_transposed_conv1d_element(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int b, int out_channel, int j,
    int L_in, int kernel_size, int stride, int padding,
    int out_channels_per_group, int in_channels_per_group, int in_channels) {
  scalar_t sum = 0;
  // Compute the range of input positions that can contribute to output index j
  int p_min = (j + padding - (kernel_size - 1) + stride - 1) / stride;  // ceiling division
  if (p_min < 0) p_min = 0;
  int p_max = (j + padding) / stride;  // integer division acts as floor for positive numbers
  if (p_max >= L_in) p_max = L_in - 1;

  // Determine group and relative channel index
  int g = out_channel / out_channels_per_group;
  int out_channel_idx = out_channel - g * out_channels_per_group;

  // Loop over input channels within the group
  for (int i = 0; i < in_channels_per_group; i++) {
    int global_in_channel = g * in_channels_per_group + i;
    // Loop over valid input spatial positions
    for (int p = p_min; p <= p_max; p++) {
      int k = j + padding - p * stride;
      if (k >= 0 && k < kernel_size) {
        // x: [N, in_channels, L_in]
        int x_idx = ((b * in_channels + global_in_channel) * L_in + p);
        // weight: [in_channels, out_channels_per_group, kernel_size]
        int w_idx = ((global_in_channel * out_channels_per_group + out_channel_idx) * kernel_size + k);
        sum += input[x_idx] * weight[w_idx];
      }
    }
  }
  return sum;
}

// CUDA kernel launching one thread per output element
template <typename scalar_t>
__global__ void conv_transposed_1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,  // can be nullptr if not provided
    scalar_t* __restrict__ output,
    int N, int in_channels, int L_in,
    int out_channels, int L_out,
    int kernel_size, int stride, int padding,
    int groups, int out_channels_per_group, int in_channels_per_group) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * out_channels * L_out;
  if (idx < total) {
    int j = idx % L_out;
    int tmp = idx / L_out;
    int out_channel = tmp % out_channels;
    int b = tmp / out_channels;

    scalar_t value = compute_transposed_conv1d_element<scalar_t>(
        input, weight, b, out_channel, j, L_in, kernel_size, stride, padding,
        out_channels_per_group, in_channels_per_group, in_channels);

    if (bias != nullptr) {
      value += bias[out_channel];
    }

    output[idx] = value;
  }
}

// Host forward function
torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  if (bias.has_value()) {
    CHECK_INPUT(bias.value());
  }

  // Get dimensions from the input tensors
  int N = x.size(0);
  int in_channels = x.size(1);
  int L_in = x.size(2);

  // weight shape is assumed to be [in_channels, out_channels_per_group, kernel_size]
  int out_channels_per_group = weight.size(1);
  int kernel_size = weight.size(2);

  int in_channels_per_group = in_channels / groups;
  int out_channels = groups * out_channels_per_group;

  // Compute output length for transposed convolution
  int L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;

  auto output = torch::empty({N, out_channels, L_out}, x.options());

  int total_elements = N * out_channels * L_out;
  int threads = 128;
  int blocks = (total_elements + threads - 1) / threads;
  
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transposed_1d_cuda", ([&] {
    const scalar_t* x_ptr = x.data_ptr<scalar_t>();
    const scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
    const scalar_t* bias_ptr = (bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr);
    scalar_t* output_ptr = output.data_ptr<scalar_t>();
    
    conv_transposed_1d_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x_ptr, weight_ptr, bias_ptr, output_ptr,
        N, in_channels, L_in,
        out_channels, L_out,
        kernel_size, stride, padding,
        groups, out_channels_per_group, in_channels_per_group);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed 1D convolution forward (CUDA) with modular device functions");
}
