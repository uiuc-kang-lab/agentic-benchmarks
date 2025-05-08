#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define tile dimensions for the output spatial domain
#define TILE_W 16
#define TILE_H 16

// CUDA kernel using shared memory to cache weight values for a fixed output channel
// Assumes blockIdx.z encodes both sample (n) and output channel (c_out): n = blockIdx.z / C_out, c_out = blockIdx.z % C_out
__global__ void shared_weight_conv_transposed_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool bias_present) {

  // Determine the output spatial position for this thread
  int w_out = blockIdx.x * TILE_W + threadIdx.x;
  int h_out = blockIdx.y * TILE_H + threadIdx.y;

  // Decode sample index and output channel from blockIdx.z
  int nc = blockIdx.z;
  int n = nc / C_out;
  int c_out = nc % C_out;

  // Return if the thread is out-of-bound for the output spatial dimensions
  if (w_out >= W_out || h_out >= H_out) return;

  // Declare external shared memory for the weight tile
  // Size needed: (C_in * kernel_h * kernel_w) floats
  extern __shared__ float shared_weight[];
  int weight_tile_size = C_in * kernel_h * kernel_w;

  // Cooperative loading of the weight tile into shared memory
  // Each thread in the block loads one or more elements, using a flat index
  int local_thread_id = threadIdx.y * TILE_W + threadIdx.x;
  int block_threads = TILE_W * TILE_H;
  for (int idx = local_thread_id; idx < weight_tile_size; idx += block_threads) {
    int c = idx / (kernel_h * kernel_w);
    int rem = idx % (kernel_h * kernel_w);
    int i = rem / kernel_w;
    int j = rem % kernel_w;
    // Global weight index: For a given input channel c and kernel position (i, j),
    // the weight is stored at: weight[c * (C_out * kernel_h * kernel_w) + c_out * (kernel_h * kernel_w) + i * kernel_w + j]
    int global_weight_index = c * (C_out * kernel_h * kernel_w) + c_out * (kernel_h * kernel_w) + i * kernel_w + j;
    shared_weight[idx] = weight[global_weight_index];
  }
  __syncthreads();

  float sum = 0.0f;

  // Loop over input channels and the kernel window
  for (int c = 0; c < C_in; c++) {
    for (int i = 0; i < kernel_h; i++) {
      for (int j = 0; j < kernel_w; j++) {
        // Compute the corresponding input coordinate for this kernel element
        int h_diff = h_out + pad_h - i * dilation_h;
        int w_diff = w_out + pad_w - j * dilation_w;
        // Check if these diffs are divisible by the stride
        if ((h_diff % stride_h == 0) && (w_diff % stride_w == 0)) {
          int h_in = h_diff / stride_h;
          int w_in = w_diff / stride_w;
          // Validate the computed input indices
          if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            int input_index = n * (C_in * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;
            float input_val = input[input_index];
            int shared_index = c * (kernel_h * kernel_w) + i * kernel_w + j;
            float weight_val = shared_weight[shared_index];
            sum += input_val * weight_val;
          }
        }
      }
    }
  }

  // Add bias if provided
  if (bias_present) {
    sum += bias[c_out];
  }

  // Write the computed sum to the output tensor
  int output_index = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
  output[output_index] = sum;
}

// Host function wrapping the CUDA kernel launch
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

  // Assume groups == 1 for simplicity
  auto N = x.size(0);
  auto C_in = x.size(1);
  auto H_in = x.size(2);
  auto W_in = x.size(3);
  auto C_out = weight.size(1);
  int kernel_h = weight.size(2);
  int kernel_w = weight.size(3);

  // Compute output dimensions using the transposed convolution formula
  int H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + output_padding[0] + 1;
  int W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + output_padding[1] + 1;

  auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

  // Set up a 3D grid: grid.x and grid.y cover the spatial tile, grid.z covers (n, c_out)
  dim3 block(TILE_W, TILE_H);
  dim3 grid((W_out + TILE_W - 1) / TILE_W,
            (H_out + TILE_H - 1) / TILE_H,
            N * C_out);

  // Calculate the required shared memory size: one block loads weights for one output channel
  size_t shared_mem_size = C_in * kernel_h * kernel_w * sizeof(float);

  bool bias_present = bias.has_value() && bias.value().defined();
  const float* bias_ptr = bias_present ? bias.value().data_ptr<float>() : nullptr;

  shared_weight_conv_transposed_kernel<<<grid, block, shared_mem_size>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias_ptr,
      output.data_ptr<float>(),
      N, C_in, H_in, W_in,
      C_out, H_out, W_out,
      kernel_h, kernel_w,
      stride[0], stride[1],
      padding[0], padding[1],
      dilation[0], dilation[1],
      bias_present);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA) with shared memory for weights");
}
