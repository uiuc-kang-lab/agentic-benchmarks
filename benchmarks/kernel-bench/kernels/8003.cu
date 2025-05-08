#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define tile dimensions for output spatial region and an input channel partition size
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define PART_SIZE 4

// Kernel: Each thread computes a partial sum over a subset of input channels (of size PART_SIZE) for a single output pixel.
// Multiple blocks (each processing a different partition of input channels) will accumulate into the same output element via atomicAdd.

__global__ void conv2d_atomic_kernel(
    const float * __restrict__ input,
    const float * __restrict__ weight,
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

  // Determine the number of partitions along the input channel dimension
  int partitions = (in_channels + PART_SIZE - 1) / PART_SIZE;

  // Decode grid dimension on z axis: each block on z covers one (batch, out_channel, partition)
  int linear_idx = blockIdx.z;
  int total_oc_parts = out_channels * partitions;
  int b = linear_idx / total_oc_parts;
  int rem = linear_idx % total_oc_parts;
  int out_ch = rem / partitions;
  int part = rem % partitions;

  // Determine the output spatial coordinate that this thread computes
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= out_height || col >= out_width) return;

  float partial_sum = 0.0f;
  
  // Calculate the origin in the input image corresponding to this output pixel
  int in_row_origin = row * stride - padding;
  int in_col_origin = col * stride - padding;

  // Each block processes a subset of input channels in the range [start_ic, end_ic)
  int start_ic = part * PART_SIZE;
  int end_ic = start_ic + PART_SIZE;
  if(end_ic > in_channels) end_ic = in_channels;

  // Loop over the assigned input channels
  for (int ic = start_ic; ic < end_ic; ++ic) {
    // Loop over the kernel window
    for (int kh = 0; kh < kernel_size; ++kh) {
      int in_r = in_row_origin + kh * dilation;
      if (in_r < 0 || in_r >= in_height) continue;
      for (int kw = 0; kw < kernel_size; ++kw) {
        int in_c = in_col_origin + kw * dilation;
        if (in_c < 0 || in_c >= in_width) continue;
        
        // Compute indices for the input and weight tensors
        int input_idx = ((b * in_channels + ic) * in_height + in_r) * in_width + in_c;
        int weight_idx = ((out_ch * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
        
        partial_sum += input[input_idx] * weight[weight_idx];
      }
    }
  }

  // Use atomicAdd to accumulate the partial sum computed for this partition
  int output_idx = ((b * out_channels + out_ch) * out_height + row) * out_width + col;
  atomicAdd(&output[output_idx], partial_sum);
}


// Host function

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
  int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

  // Allocate output tensor and pre-fill with bias if provided
  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
  if (bias.has_value()) {
    // Broadcast bias to the output tensor
    output.copy_(bias.value().view({1, out_channels, 1, 1}).expand({batch_size, out_channels, out_height, out_width}));
  }

  // Determine the number of partitions
  int partitions = (in_channels + PART_SIZE - 1) / PART_SIZE;

  // Set up block and grid dimensions
  dim3 block(TILE_WIDTH, TILE_HEIGHT);
  dim3 grid(
      (out_width + TILE_WIDTH - 1) / TILE_WIDTH,
      (out_height + TILE_HEIGHT - 1) / TILE_HEIGHT,
      batch_size * out_channels * partitions);

  conv2d_atomic_kernel<<<grid, block>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
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
  m.def("forward", &forward, "2D convolution using input channel partitioning with minimal atomic operations");
}
