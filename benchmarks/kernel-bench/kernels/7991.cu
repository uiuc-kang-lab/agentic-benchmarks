#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare constant memory for weights and bias (adjust sizes as needed)
// Here we assume maximum sizes that fit typical small kernels (e.g. kernel size 3-7).
__constant__ float d_weight[12288];  // adjust size if needed
__constant__ float d_bias[1024];     // adjust size if needed

// Kernel using grid-stride loops to evenly distribute the workload across threads
__global__ void conv2d_even_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width,
    int use_bias) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int gridSize = blockDim.x * gridDim.x;

  // Each thread processes multiple output elements via a grid-stride loop
  for (int index = tid; index < total_output; index += gridSize) {
    int tmp = index;
    int col = tmp % out_width;
    tmp /= out_width;
    int row = tmp % out_height;
    tmp /= out_height;
    int channel = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = 0.0f;
    int in_row_origin = row * stride - padding;
    int in_col_origin = col * stride - padding;

    // Loop over all input channels and kernel elements
    for (int ic = 0; ic < in_channels; ++ic) {
      for (int kh = 0; kh < kernel_size; ++kh) {
        int in_row = in_row_origin + kh * dilation;
        if (in_row < 0 || in_row >= in_height) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
          int in_col = in_col_origin + kw * dilation;
          if (in_col < 0 || in_col >= in_width) continue;
          int input_idx = ((b * in_channels + ic) * in_height + in_row) * in_width + in_col;
          int weight_idx = ((channel * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
          float in_val = __ldg(&input[input_idx]);
          float wt_val = d_weight[weight_idx];
          sum += in_val * wt_val;
        }
      }
    }
    if (use_bias) {
      sum += d_bias[channel];
    }
    output[index] = sum;
  }
}


// Host function to launch the kernel
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
  TORCH_CHECK(groups == 1, "even_dist_conv2d_kernel only supports groups == 1");

  // Input dimensions
  int batch_size = x.size(0);
  int in_channels = x.size(1);
  int in_height = x.size(2);
  int in_width = x.size(3);

  // Weight dimensions (assumes square kernel and standard conv layout)
  int out_channels = weight.size(0);
  int kernel_size = weight.size(2);

  // Compute output dimensions
  int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
  int out_width  = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

  int total_output = batch_size * out_channels * out_height * out_width;

  // Copy weights to constant memory
  cudaMemcpyToSymbol(d_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float));
  int use_bias = 0;
  if (bias.has_value()) {
    cudaMemcpyToSymbol(d_bias, bias.value().data_ptr<float>(), bias.value().numel() * sizeof(float));
    use_bias = 1;
  }

  // Launch the kernel with grid-stride loop distribution
  int blockSize = 256;
  int gridSize = (total_output + blockSize - 1) / blockSize;
  conv2d_even_kernel<<<gridSize, blockSize>>>(
      x.data_ptr<float>(),
      output.data_ptr<float>(),
      total_output,
      batch_size,
      in_channels,
      in_height,
      in_width,
      out_channels,
      kernel_size,
      stride,
      padding,
      dilation,
      out_height,
      out_width,
      use_bias
  );

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Evenly Distributed 2D convolution with constant memory and grid-stride loop");
}
