#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel minimizes warp divergence by precomputing the valid pooling window indices
// so that the inner loop does not perform in-loop boundary checks.

__global__ void max_pool1d_no_divergence_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int num_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length,
    const bool return_indices) {

  // Compute flat thread index over the entire output tensor
  int total = batch_size * num_channels * output_length;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) return;

  // Map flat index to (b, c, i) coordinates
  int i = tid % output_length;
  int bc = tid / output_length;
  int b = bc / num_channels;
  int c = bc % num_channels;

  // Compute the starting index of the pooling window in the input
  int input_start = i * stride - padding;

  // Precompute the valid range for the pooling window indices to avoid in-loop bounds checks.
  // k_min is set to the smallest k such that input_start + k*dilation >= 0.
  int k_min = (input_start < 0) ? ((-input_start + dilation - 1) / dilation) : 0;
  
  // k_max is the smallest k for which input_start + k*dilation reaches or exceeds input_length.
  int k_max = (input_length - input_start + dilation - 1) / dilation; // effectively ceil((input_length - input_start)/dilation)
  if (k_max > kernel_size) k_max = kernel_size;

  // Initialize with first valid element to avoid the branch in the first iteration
  int first_pos = input_start + k_min * dilation;
  float max_val = __ldg(input + b * num_channels * input_length + c * input_length + first_pos);
  int max_idx = first_pos;

  // Loop over remaining elements in the valid pooling window
  #pragma unroll
  for (int k = k_min + 1; k < k_max; k++) {
    int pos = input_start + k * dilation;
    float val = __ldg(input + b * num_channels * input_length + c * input_length + pos);
    // Use fmaxf instead of if statement to reduce branching
    if (val > max_val) {
      max_val = val;
      max_idx = pos;
    }
  }

  // Write results, ensuring coalesced accesses on output
  int out_idx = b * num_channels * output_length + c * output_length + i;
  output[out_idx] = max_val;
  if (return_indices) {
    indices[out_idx] = max_idx;
  }
}


// Forward function to set up kernel parameters and launch the kernel
torch::Tensor forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {
  TORCH_CHECK(x.dim() == 3, "Input must be a 3D tensor.");
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous.");

  const int batch_size = x.size(0);
  const int num_channels = x.size(1);
  const int input_length = x.size(2);
  
  // Compute output length as in standard convolution/pooling operations
  const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
  TORCH_CHECK(output_length > 0, "Output length must be positive.");
  
  auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  auto output = torch::empty({batch_size, num_channels, output_length}, options);
  torch::Tensor indices;
  if (return_indices) {
    indices = torch::empty({batch_size, num_channels, output_length}, options.dtype(torch::kInt64));
  }

  int total = batch_size * num_channels * output_length;
  const int threads_per_block = 256;
  const int num_blocks = (total + threads_per_block - 1) / threads_per_block;

  max_pool1d_no_divergence_kernel<<<num_blocks, threads_per_block>>>(
      x.data_ptr<float>(),
      output.data_ptr<float>(),
      return_indices ? indices.data_ptr<int64_t>() : nullptr,
      batch_size,
      num_channels,
      input_length,
      kernel_size,
      stride,
      padding,
      dilation,
      output_length,
      return_indices
  );

  return return_indices ? torch::cat({output, indices}, -1) : output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "MaxPool1D forward with minimized warp divergence (CUDA)");
}
