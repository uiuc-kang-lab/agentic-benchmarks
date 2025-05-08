#include <torch/extension.h>
#include <cuda.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

// This kernel uses shared memory and warp-level primitives to perform the reduction
// of the convolution summation. Each block is responsible for computing one output element.
// The reduction over the combined kernel spatial dimensions and input-channel block is split
// across threads, then reduced using __shfl_down_sync and shared memory, which minimizes
// the cost of the reduction operations.

__global__ void transposed_conv3d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr
    float* __restrict__ output,
    int batch,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int out_d,
    int out_h,
    int out_w,
    int k_d,
    int k_h,
    int k_w,
    int s_d,
    int s_h,
    int s_w,
    int p_d,
    int p_h,
    int p_w,
    int groups,
    int channels_per_group_in,
    int channels_per_group_out) {

  // Each block computes one output element. Decode blockIdx.x into (n, oc, d_out, h_out, w_out).
  int idx = blockIdx.x;
  int w_out = idx % out_w; idx /= out_w;
  int h_out = idx % out_h; idx /= out_h;
  int d_out = idx % out_d; idx /= out_d;
  int oc = idx % out_channels; idx /= out_channels;
  int n = idx;  

  // Compute base coordinates (with padding added) in the input space
  int d_base = d_out + p_d;
  int h_base = h_out + p_h;
  int w_base = w_out + p_w;

  float partial_sum = 0.0f;

  // Reduction dimension: combine kernel spatial elements and input channel slice
  int reduction_size = channels_per_group_in * (k_d * k_h * k_w);

  // Each thread in the block processes a subset of the reduction elements
  for (int r = threadIdx.x; r < reduction_size; r += blockDim.x) {
      int ic = r / (k_d * k_h * k_w);
      int rem = r % (k_d * k_h * k_w);
      int kd = rem / (k_h * k_w);
      rem = rem % (k_h * k_w);
      int kh = rem / k_w;
      int kw = rem % k_w;

      // Determine group indices
      int group = oc / channels_per_group_out;
      int oc_in_group = oc % channels_per_group_out;

      // Compute the corresponding input indices
      int tmp_d = d_base - kd;
      if (tmp_d % s_d != 0) continue;
      int in_d_idx = tmp_d / s_d;
      if (in_d_idx < 0 || in_d_idx >= in_d) continue;

      int tmp_h = h_base - kh;
      if (tmp_h % s_h != 0) continue;
      int in_h_idx = tmp_h / s_h;
      if (in_h_idx < 0 || in_h_idx >= in_h) continue;

      int tmp_w = w_base - kw;
      if (tmp_w % s_w != 0) continue;
      int in_w_idx = tmp_w / s_w;
      if (in_w_idx < 0 || in_w_idx >= in_w) continue;

      int in_channel = group * channels_per_group_in + ic;

      // Compute flattened index for the input tensor
      int input_idx = n * (in_channels * in_d * in_h * in_w) +
                      in_channel * (in_d * in_h * in_w) +
                      in_d_idx * (in_h * in_w) +
                      in_h_idx * in_w +
                      in_w_idx;

      // Compute flattened index for the weight tensor
      int weight_idx = in_channel * (channels_per_group_out * k_d * k_h * k_w) +
                       oc_in_group * (k_d * k_h * k_w) +
                       kd * (k_h * k_w) +
                       kh * k_w +
                       kw;

      partial_sum += input[input_idx] * weight[weight_idx];
  }

  // Warp-level reduction using __shfl_down_sync
  unsigned int mask = 0xffffffff;
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
      partial_sum += __shfl_down_sync(mask, partial_sum, offset);
  }

  // Shared memory reduction among warps
  __shared__ float shared_sum[32];  // enough for up to 1024 threads (32 warps)
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  if (lane == 0) {
      shared_sum[warp_id] = partial_sum;
  }
  __syncthreads();

  float block_sum = 0.0f;
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  if (threadIdx.x == 0) {
      for (int i = 0; i < num_warps; i++) {
          block_sum += shared_sum[i];
      }
      // Add bias if provided
      if (bias != nullptr) {
          block_sum += bias[oc];
      }
      output[blockIdx.x] = block_sum;
  }
}

// Forward function wrapping the custom CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,  // not used in kernel, assumed computed externally
    int64_t groups) {

  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  if (bias_opt.has_value()) {
      CHECK_INPUT(*bias_opt);
  }

  // Extract input dimensions
  int batch = x.size(0);
  int in_channels = x.size(1);
  int in_d = x.size(2);
  int in_h = x.size(3);
  int in_w = x.size(4);

  // Extract kernel dimensions (assumed shape: [in_channels, out_channels_per_group, k_d, k_h, k_w])
  int k_d = weight.size(2);
  int k_h = weight.size(3);
  int k_w = weight.size(4);

  // Stride and padding
  int s_d = stride[0];
  int s_h = stride[1];
  int s_w = stride[2];
  int p_d = padding[0];
  int p_h = padding[1];
  int p_w = padding[2];
  int op_d = output_padding[0];
  int op_h = output_padding[1];
  int op_w = output_padding[2];

  // Compute output dimensions using transposed convolution formula
  int out_d = (in_d - 1) * s_d - 2 * p_d + k_d + op_d;
  int out_h = (in_h - 1) * s_h - 2 * p_h + k_h + op_h;
  int out_w = (in_w - 1) * s_w - 2 * p_w + k_w + op_w;

  // Determine output channels
  int channels_per_group_out = weight.size(1);
  int out_channels = channels_per_group_out * groups;
  int channels_per_group_in = in_channels / groups;

  auto output = torch::empty({batch, out_channels, out_d, out_h, out_w}, x.options());

  // Total number of output elements. Each block computes one output element.
  int total_outputs = batch * out_channels * out_d * out_h * out_w;

  // Launch parameters: use a block size that maps well to warp-level reduction
  int block_size = 128;
  int grid_size = total_outputs;

  const float* x_ptr = x.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const float* bias_ptr = bias_opt.has_value() ? (*bias_opt).data_ptr<float>() : nullptr;
  float* out_ptr = output.data_ptr<float>();

  transposed_conv3d_shared_kernel<<<grid_size, block_size>>>(
      x_ptr,
      weight_ptr,
      bias_ptr,
      out_ptr,
      batch,
      in_channels,
      in_d,
      in_h,
      in_w,
      out_channels,
      out_d,
      out_h,
      out_w,
      k_d,
      k_h,
      k_w,
      s_d,
      s_h,
      s_w,
      p_d,
      p_h,
      p_w,
      groups,
      channels_per_group_in,
      channels_per_group_out
  );

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward with shared memory reduction (CUDA)");
}
