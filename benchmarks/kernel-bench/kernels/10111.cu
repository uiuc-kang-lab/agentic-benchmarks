#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Define block size for thread-level parallelism
#define BLOCK_SIZE 256

// Fused depthwise separable convolution kernel with atomic accumulation.
// This kernel fuses the depthwise and pointwise convolutions. It distributes the summation
// over the input channels (depthwise channels) across multiple thread blocks.
// Each block computes a partial sum over a chunk of the input channels, reduces it in shared
// memory, and then uses a single atomicAdd to accumulate into the global output. Additionally,
// the pointwise bias is added only from the first block (blockIdx.y == 0) to avoid multiple bias additions.

template <typename scalar_t>
__global__ void fused_conv_atomic_kernel(
    const scalar_t* __restrict__ input,           // [batch, in_channels, in_h, in_w]
    const scalar_t* __restrict__ depthwise_weight,  // [in_channels, 1, k, k] stored as [in_channels * k * k]
    const scalar_t* __restrict__ depthwise_bias,    // [in_channels] or nullptr
    const scalar_t* __restrict__ pointwise_weight,  // [out_channels, in_channels] stored row-major
    const scalar_t* __restrict__ pointwise_bias,    // [out_channels] or nullptr
    scalar_t* __restrict__ output,                  // [batch, out_channels, out_h, out_w]
    int batch,
    int in_channels,
    int in_h, int in_w,
    int out_channels,
    int out_h, int out_w,
    int k,            // kernel size (assumed square kernel k x k)
    int stride,
    int padding,
    int dilation) {

  // Calculate the flat index for the output element this block will contribute to
  int out_idx = blockIdx.x;
  int total_outputs = batch * out_channels * out_h * out_w;
  if (out_idx >= total_outputs) return;

  // Decode flat index into (n, oc, oh, ow)
  int ow = out_idx % out_w;
  int tmp = out_idx / out_w;
  int oh = tmp % out_h;
  tmp = tmp / out_h;
  int oc = tmp % out_channels;
  int n = tmp / out_channels;

  // Each block in the y-dimension covers a chunk of the input channels (ic)
  // Calculate the starting index for this block's chunk in the input channel dimension
  int block_ic_offset = blockIdx.y * blockDim.x;  // Offset for input channel index
  int threadId = threadIdx.x;
  // The stride over ic across all blocks in grid.y
  int ic_stride = gridDim.y * blockDim.x;

  scalar_t local_sum = 0;

  // Loop over input channels assigned to this thread
  for (int ic = block_ic_offset + threadId; ic < in_channels; ic += ic_stride) {
    // Compute the depthwise convolution for input channel ic at output spatial position (oh, ow)
    scalar_t d_sum = 0;
    // Loop over the k x k kernel window
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < k; j++) {
        int ih = oh * stride - padding + i * dilation;
        int iw = ow * stride - padding + j * dilation;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
          int input_idx = n * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
          int weight_idx = ic * (k * k) + i * k + j;
          d_sum += input[input_idx] * depthwise_weight[weight_idx];
        }
      }
    }
    if (depthwise_bias != nullptr) {
      d_sum += depthwise_bias[ic];
    }
    // Multiply the depthwise output by the corresponding pointwise weight
    // pointwise_weight is stored as [out_channels, in_channels]
    int pw_idx = oc * in_channels + ic;
    local_sum += d_sum * pointwise_weight[pw_idx];
  }

  // Perform block-level reduction in shared memory
  __shared__ volatile scalar_t shared_data[BLOCK_SIZE];
  shared_data[threadId] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadId < s) {
      shared_data[threadId] += shared_data[threadId + s];
    }
    __syncthreads();
  }

  // Thread 0 of each block writes its partial sum to the global output using atomicAdd
  if (threadId == 0) {
    atomicAdd(&output[out_idx], shared_data[0]);
    // Add the pointwise bias only once per output element (from the first ic chunk block)
    if (blockIdx.y == 0 && pointwise_bias != nullptr) {
      atomicAdd(&output[out_idx], pointwise_bias[oc]);
    }
  }
}

// The forward CUDA function that sets up and launches the fused kernel

torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride,
    int padding,
    int dilation) {

  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
  TORCH_CHECK(depthwise_weight.is_cuda(), "Depthwise weight must be a CUDA tensor");
  TORCH_CHECK(pointwise_weight.is_cuda(), "Pointwise weight must be a CUDA tensor");
  if (depthwise_bias.defined() && depthwise_bias.numel() > 0)
    TORCH_CHECK(depthwise_bias.is_cuda(), "Depthwise bias must be a CUDA tensor if provided");
  if (pointwise_bias.defined() && pointwise_bias.numel() > 0)
    TORCH_CHECK(pointwise_bias.is_cuda(), "Pointwise bias must be a CUDA tensor if provided");

  int batch = x.size(0);
  int in_channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);

  // Depthwise weight shape: [in_channels, 1, k, k]
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  // Pointwise weight shape: [out_channels, in_channels]
  int out_channels = pointwise_weight.size(0);

  // Allocate output tensor and initialize to 0 (atomic accumulations require a zero-initialized array)
  auto output = torch::zeros({batch, out_channels, out_h, out_w}, x.options());

  // Total number of output elements
  int total_outputs = batch * out_channels * out_h * out_w;

  // Determine grid dimensions:
  // grid.x covers each output element, and grid.y splits the in_channels dimension
  int grid_y = (in_channels + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid(total_outputs, grid_y);
  dim3 block(BLOCK_SIZE);

  // Launch the kernel using AT_DISPATCH for the correct scalar type
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_conv_atomic_cuda", ([&] {
    fused_conv_atomic_kernel<scalar_t><<<grid, block>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        depthwise_bias.defined() && depthwise_bias.numel() > 0 ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
        pointwise_weight.data_ptr<scalar_t>(),
        pointwise_bias.defined() && pointwise_bias.numel() > 0 ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
        output.data_ptr<scalar_t>(),
        batch,
        in_channels,
        in_h, in_w,
        out_channels,
        out_h, out_w,
        k,
        stride,
        padding,
        dilation);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Fused kernel launch error: %s\n", cudaGetErrorString(err));
  }
  return output;
}

// Helper: convert a py::object to a tensor
at::Tensor toTensor(const py::object& obj) {
  if (obj.is_none()) {
    return at::Tensor();
  }
  try {
    return obj.cast<at::Tensor>();
  } catch (const py::cast_error& e) {
    if (py::hasattr(obj, "data")) {
      return obj.attr("data").cast<at::Tensor>();
    }
    throw std::runtime_error("Expected a torch Tensor or Parameter.");
  }
}

// Wrapper function to allow flexible inputs: it accepts Tensors or Parameters; if None, uses undefined tensor.
// Expected signature: forward(tensor, tensor, tensor, tensor, tensor, int, int, int) → tensor
at::Tensor forward_wrapper(py::object x_obj,
                           py::object depthwise_weight_obj,
                           py::object pointwise_weight_obj,
                           py::object depthwise_bias_obj,
                           py::object pointwise_bias_obj,
                           int stride,
                           int padding,
                           int dilation) {
  auto x = toTensor(x_obj);
  auto depthwise_weight = toTensor(depthwise_weight_obj);
  auto pointwise_weight = toTensor(pointwise_weight_obj);
  auto depthwise_bias = toTensor(depthwise_bias_obj);
  auto pointwise_bias = toTensor(pointwise_bias_obj);

  return forward_cuda(x, depthwise_weight, pointwise_weight,
                      depthwise_bias, pointwise_bias,
                      stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper, "Fused CUDA depthwise separable convolution forward with minimal atomic usage");
}
