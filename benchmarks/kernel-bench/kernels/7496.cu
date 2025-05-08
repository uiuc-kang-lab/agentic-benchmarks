#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel function that processes a chunk of the batch
__global__ void convTranspose2DKernelStream(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr
    float* __restrict__ output,
    int batch_offset,    // starting index for this chunk
    int sub_batch,       // number of batches in this chunk
    int in_channels,
    int out_channels,
    int H_in,
    int W_in,
    int kernel_size,
    int stride,
    int padding,
    int H_out,
    int W_out) {

  // total number of output elements in this batch chunk
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = sub_batch * out_channels * H_out * W_out;
  if (idx >= total) return;

  // Decode linear index into (n_local, oc, h, w)
  int w = idx % W_out;
  int tmp = idx / W_out;
  int h = tmp % H_out;
  tmp /= H_out;
  int oc = tmp % out_channels;
  int n_local = tmp / out_channels;
  int n = batch_offset + n_local;  // global batch index

  float value = 0.0f;
  
  // Compute convolution transpose sum
  for (int ic = threadIdx.x % in_channels; ic < in_channels; ic += blockDim.x) {
    for (int p = 0; p < kernel_size; p++) {
      int h_offset = h + padding - p;
      if (h_offset < 0 || (h_offset % stride != 0)) continue;
      int i_in = h_offset / stride;
      if (i_in < 0 || i_in >= H_in) continue;
      for (int q = 0; q < kernel_size; q++) {
        int w_offset = w + padding - q;
        if (w_offset < 0 || (w_offset % stride != 0)) continue;
        int j_in = w_offset / stride;
        if (j_in < 0 || j_in >= W_in) continue;
        int input_index = ((n * in_channels + ic) * H_in + i_in) * W_in + j_in;
        int weight_index = ((ic * out_channels + oc) * kernel_size + p) * kernel_size + q;
        value += __ldg(&input[input_index]) * __ldg(&weight[weight_index]);
      }
    }
  }
  
  if (bias) {
    value += __ldg(&bias[oc]);
  }

  int output_index = ((n * out_channels + oc) * H_out + h) * W_out + w;
  output[output_index] = value;
}

// Host function that uses CUDA streams to overlap kernel execution across batch chunks

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
  TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
  TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
  TORCH_CHECK(groups == 1, "This implementation supports groups==1 only.");

  bool bias_present = false;
  torch::Tensor bias_tensor;
  if (bias.has_value()) {
    bias_tensor = bias.value();
    TORCH_CHECK(bias_tensor.is_cuda(), "Bias tensor must be on CUDA");
    TORCH_CHECK(bias_tensor.is_contiguous(), "Bias tensor must be contiguous");
    bias_present = true;
  }

  // Dimensions
  int batch = x.size(0);
  int in_channels = x.size(1);
  int H_in = x.size(2);
  int W_in = x.size(3);
  int kernel_size = weight.size(2);  // square kernel assumed, weight shape: [in_channels, out_channels, k, k]
  int out_channels = weight.size(1);

  // Compute output spatial dimensions
  int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
  int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

  auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  torch::Tensor output = torch::empty({batch, out_channels, H_out, W_out}, options);

  // Decide on number of streams; use up to 4 streams or fewer if batch is small
  int num_streams = (batch < 4) ? batch : 4;
  int chunk = (batch + num_streams - 1) / num_streams; // round-up division

  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  int block_size = 256;
  
  // Launch kernel for each batch chunk on separate streams
  for (int i = 0; i < num_streams; i++) {
    int batch_offset = i * chunk;
    if (batch_offset >= batch) break;
    int sub_batch = std::min(chunk, batch - batch_offset);

    int total_threads = sub_batch * out_channels * H_out * W_out;
    int grid_size = (total_threads + block_size - 1) / block_size;

    convTranspose2DKernelStream<<<grid_size, block_size, 0, streams[i]>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_present ? bias_tensor.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_offset,
        sub_batch,
        in_channels,
        out_channels,
        H_in,
        W_in,
        kernel_size,
        stride,
        padding,
        H_out,
        W_out
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
  }

  // Synchronize all streams
  for (int i = 0; i < num_streams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward with CUDA stream overlap (CUDA)");
}
