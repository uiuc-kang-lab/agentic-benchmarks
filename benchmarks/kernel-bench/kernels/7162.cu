#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel performs a 2D convolution where the input, weight, and bias (if provided) are read-only.
// It uses __ldg() to load data from global memory with the read-only cache and assumes that the underlying
// memory is aligned to 128-bit boundaries. The kernel supports grouped convolution. Each thread computes one output
// element, while loops over the appropriate kernel window and input channels for the group.

template <typename scalar_t>
__global__ void conv2d_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * out_channels * out_height * out_width;
  if (index < total) {
    // Calculate output coordinates
    int w = index % out_width;
    int temp = index / out_width;
    int h = temp % out_height;
    temp /= out_height;
    int oc = temp % out_channels;
    int b = temp / out_channels;

    scalar_t sum = 0;

    // Determine channel range for the current group
    int group_in_channels = in_channels / groups;
    int group_out_channels = out_channels / groups;
    int group_id = oc / group_out_channels;
    int ic_start = group_id * group_in_channels;
    int ic_end = ic_start + group_in_channels;

    // Iterate over the input channels of the group and the kernel window
    for (int ic = ic_start; ic < ic_end; ic++) {
      for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
          int ih = h * stride - padding + kh * dilation;
          int iw = w * stride - padding + kw * dilation;
          if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
            int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
            // weight is stored as [out_channels, group_in_channels, kernel_h, kernel_w]
            int weight_idx = (((oc) * group_in_channels + (ic - ic_start)) * kernel_h + kh) * kernel_w + kw;
            // Use __ldg for read-only loads (assumes 128-bit alignment provided by contiguous tensors)
            sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
          }
        }
      }
    }

    // Add bias if provided
    if (bias != nullptr) {
      sum += __ldg(&bias[oc]);
    }

    output[index] = sum;
  }
}

// Host function that prepares and launches the CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  if (bias.has_value()) {
    CHECK_INPUT(bias.value());
  }

  auto batch = x.size(0);
  auto in_channels = x.size(1);
  auto in_height = x.size(2);
  auto in_width = x.size(3);

  auto out_channels = weight.size(0);
  auto kernel_h = weight.size(2);
  auto kernel_w = weight.size(3);

  // Compute output dimensions (assuming standard convolution formula)
  auto out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
  auto out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

  auto output = torch::empty({batch, out_channels, out_height, out_width}, x.options());

  int total = batch * out_channels * out_height * out_width;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv2d_cuda_kernel", ([&] {
    conv2d_cuda_kernel<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
        output.data_ptr<scalar_t>(),
        batch,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_h,
        kernel_w,
        out_height,
        out_width,
        stride,
        padding,
        dilation,
        groups);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized CUDA forward function for 2D convolution using __ldg and aligned accesses.");
}
