#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Constant memory buffers for weights and bias
// (Assumes the weight and bias sizes fit within constant memory limits on the device)
__constant__ float d_weight[12288];
__constant__ float d_bias[1024];

// Kernel using stride loops to cover workloads larger than the number of threads available
__global__ void conv2d_const_stride_kernel(
    const float * __restrict__ input,
    float * __restrict__ output,
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
    int out_width) {

  // Each block in z-dimension corresponds to a combination of batch index and output channel
  int bc = blockIdx.z;      // combined index: bc = b * out_channels + channel
  int b = bc / out_channels;
  int channel = bc % out_channels;
  
  // Determine starting output coordinate for each thread
  int thread_col = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_row = blockIdx.y * blockDim.y + threadIdx.y;
  
  // Compute the overall stride for iterating over output space
  int col_stride = blockDim.x * gridDim.x;
  int row_stride = blockDim.y * gridDim.y;
  
  // Loop over output rows and columns using stride loops
  for (int out_row = thread_row; out_row < out_height; out_row += row_stride) {
    for (int out_col = thread_col; out_col < out_width; out_col += col_stride) {
      float sum = 0.0f;
      
      // Calculate the origin in the input
      int in_row_origin = out_row * stride - padding;
      int in_col_origin = out_col * stride - padding;
      
      // Loop over the input channels and kernel dimensions
      for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
          int in_row = in_row_origin + kh * dilation;
          // Check row boundary
          if (in_row < 0 || in_row >= in_height) continue;
          
          for (int kw = 0; kw < kernel_size; ++kw) {
            int in_col = in_col_origin + kw * dilation;
            // Check column boundary
            if (in_col < 0 || in_col >= in_width) continue;
            
            int input_idx = ((b * in_channels + ic) * in_height + in_row) * in_width + in_col;
            int weight_idx = ((channel * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
            float in_val = __ldg(&input[input_idx]);
            float wt_val = d_weight[weight_idx];
            sum += in_val * wt_val;
          }
        }
      }
      
      // Add bias from constant memory
      sum += d_bias[channel];
      
      int output_idx = ((b * out_channels + channel) * out_height + out_row) * out_width + out_col;
      output[output_idx] = sum;
    }
  }
}


// Host function to launch the CUDA kernel

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

  // Get dimensions from the input tensor
  int batch_size = x.size(0);
  int in_channels = x.size(1);
  int in_height = x.size(2);
  int in_width = x.size(3);

  // Assuming weight has shape [out_channels, in_channels, kernel_size, kernel_size]
  int out_channels = weight.size(0);
  int kernel_size = weight.size(2);

  // Compute output dimensions
  int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
  int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

  // Copy weight to constant memory
  cudaMemcpyToSymbol(d_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float));

  // Copy bias to constant memory (if not provided, use zeros)
  if (bias.has_value()) {
    cudaMemcpyToSymbol(d_bias, bias.value().data_ptr<float>(), bias.value().numel() * sizeof(float));
  } else {
    auto zero_bias = torch::zeros({out_channels}, x.options());
    cudaMemcpyToSymbol(d_bias, zero_bias.data_ptr<float>(), zero_bias.numel() * sizeof(float));
  }

  // Set up block and grid dimensions
  dim3 block(16, 16);
  dim3 grid((out_width + block.x - 1) / block.x,
            (out_height + block.y - 1) / block.y,
            batch_size * out_channels);

  conv2d_const_stride_kernel<<<grid, block>>>(
      x.data_ptr<float>(),
      output.data_ptr<float>(),
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
      out_width);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "2D convolution using constant memory and stride loops for large workloads");
}
