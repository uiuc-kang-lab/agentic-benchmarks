#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define warp size constant
constexpr int WARP_SIZE = 32;

// Optimized CUDA kernel for 2D convolution using shared memory and warp-level reduction
__global__ void conv2d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
    float* __restrict__ output,
    const int N,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation,
    const int groups) {

  // Each block computes one output element in the output tensor
  // Compute output indices from blockIdx.x
  int index = blockIdx.x;
  int ow = index % out_width;
  int oh = (index / out_width) % out_height;
  int oc = (index / (out_width * out_height)) % out_channels;
  int n = index / (out_width * out_height * out_channels);

  float sum = 0.0f;
  
  // Determine reduction loop length and input channel offset in case groups>1
  int num_k = 0;
  int in_ch_offset = 0;
  if (groups == 1) {
    num_k = in_channels * kernel_size * kernel_size;
  } else {
    int in_channels_per_group = in_channels / groups;
    num_k = in_channels_per_group * kernel_size * kernel_size;
    int out_channels_per_group = out_channels / groups;
    int group = oc / out_channels_per_group;
    in_ch_offset = group * in_channels_per_group;
  }

  // Each thread in the block computes a partial sum for the output pixel
  for (int i = threadIdx.x; i < num_k; i += blockDim.x) {
    int ic_local = i / (kernel_size * kernel_size);
    int rem = i % (kernel_size * kernel_size);
    int kH = rem / kernel_size;
    int kW = rem % kernel_size;

    int ic = (groups == 1) ? ic_local : (in_ch_offset + ic_local);

    int in_y = oh * stride - padding + kH * dilation;
    int in_x = ow * stride - padding + kW * dilation;

    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
      int input_idx = ((n * in_channels + ic) * in_height + in_y) * in_width + in_x;
      int weight_idx;
      if (groups == 1) {
        weight_idx = ((oc * in_channels + ic) * kernel_size + kH) * kernel_size + kW;
      } else {
        int in_channels_per_group = in_channels / groups;
        weight_idx = ((oc * in_channels_per_group + ic_local) * kernel_size + kH) * kernel_size + kW;
      }
      sum += input[input_idx] * weight[weight_idx];
    }
  }

  // Intra-warp reduction using warp-level shuffles
  unsigned int mask = 0xffffffff;
  for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(mask, sum, offset);
  }

  // Shared memory to store each warp's partial sum
  extern __shared__ float shared_data[];
  int lane = threadIdx.x % WARP_SIZE;
  int warpId = threadIdx.x / WARP_SIZE;
  if (lane == 0) {
    shared_data[warpId] = sum;
  }
  __syncthreads();

  // Let the first few threads reduce the warp sums
  int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  float final_sum = 0.0f;
  if (threadIdx.x < numWarps) {
    final_sum = shared_data[threadIdx.x];
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
      final_sum += __shfl_down_sync(mask, final_sum, offset);
    }
    if (threadIdx.x == 0) {
      if (bias != nullptr) {
        final_sum += bias[oc];
      }
      output[index] = final_sum;
    }
  }
}


// Host function that sets up and launches the optimized CUDA kernel
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

  // Get dimensions from the input and weight tensors
  int N = x.size(0);
  int in_channels = x.size(1);
  int in_height = x.size(2);
  int in_width = x.size(3);

  int out_channels = weight.size(0);
  int kernel_size = weight.size(2);  // square kernel assumed
  // Compute output spatial dimensions
  int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
  int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

  auto output = torch::zeros({N, out_channels, out_height, out_width}, x.options());

  // Total number of output elements equals N * out_channels * out_height * out_width
  int total_outputs = N * out_channels * out_height * out_width;

  // Configure block and grid sizes
  int threads = 256;
  int blocks = total_outputs;
  int warps_per_block = (threads + WARP_SIZE - 1) / WARP_SIZE;
  int sharedMemBytes = warps_per_block * sizeof(float);

  // Launch the CUDA kernel
  conv2d_optimized_kernel<<<blocks, threads, sharedMemBytes>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
      output.data_ptr<float>(),
      N,
      in_channels,
      in_height,
      in_width,
      out_channels,
      kernel_size,
      out_height,
      out_width,
      stride,
      padding,
      dilation,
      groups);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized CUDA forward function for 2D convolution with warp-level reduction");
}
