#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

// This kernel uses one block per output element. Each thread in the block computes a partial sum
// (over a subset of the reduction iteration space: kernel depth*height*width multiplied by input channels per group),
// stores the result in shared memory, and then a block-level reduction is performed in two stages:
// first using shared memory and then finishing with warp-level primitives (__shfl_down_sync()).
// Finally, if a bias is provided, it is added to the reduced sum in thread 0 before writing the output.

__global__ void transposed_conv3d_shared_reduction_kernel(
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

  // Each block computes one output element
  int out_idx = blockIdx.x; 
  int total_outputs = batch * out_channels * out_d * out_h * out_w;
  if (out_idx >= total_outputs) return;

  // Decode out_idx into coordinates: order assumed [n, oc, d_out, h_out, w_out]
  int tmp = out_idx;
  int w_out_coord = tmp % out_w; tmp /= out_w;
  int h_out_coord = tmp % out_h; tmp /= out_h;
  int d_out_coord = tmp % out_d; tmp /= out_d;
  int oc = tmp % out_channels; tmp /= out_channels;
  int n = tmp;

  // Compute base indices for the output location (incorporate padding)
  int d_base = d_out_coord + p_d;
  int h_base = h_out_coord + p_h;
  int w_base = w_out_coord + p_w;

  // Determine group and intra-group output channel
  int group = oc / channels_per_group_out;
  int oc_in_group = oc % channels_per_group_out;

  // Total reduction iterations: iterate over kernel elements and input channels within the group
  int total_iter = k_d * k_h * k_w * channels_per_group_in;

  float sum = 0.0f;
  // Each thread processes a subset of the reduction loop, striding over blockDim.x
  for (int i = threadIdx.x; i < total_iter; i += blockDim.x) {
    int ic = i % channels_per_group_in;
    int tmp_i = i / channels_per_group_in;
    int kd = tmp_i / (k_h * k_w);
    int rem = tmp_i % (k_h * k_w);
    int kh = rem / k_w;
    int kw = rem % k_w;

    // Compute corresponding input coordinate for this kernel position
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
    // Compute flattened index for input: [n, in_channel, in_d, in_h, in_w]
    int input_idx = n * (in_channels * in_d * in_h * in_w) +
                    in_channel * (in_d * in_h * in_w) +
                    in_d_idx * (in_h * in_w) +
                    in_h_idx * in_w +
                    in_w_idx;
    float in_val = input[input_idx];

    // Compute flattened index for weight: [in_channel, oc_in_group, k_d, k_h, k_w]
    int weight_idx = in_channel * (channels_per_group_out * k_d * k_h * k_w) +
                     oc_in_group * (k_d * k_h * k_w) +
                     kd * (k_h * k_w) +
                     kh * k_w +
                     kw;
    float wt_val = weight[weight_idx];

    sum += in_val * wt_val;
  }

  // Allocate shared memory for block-level reduction
  extern __shared__ float sdata[];
  sdata[threadIdx.x] = sum;
  __syncthreads();

  // Intra-block reduction in shared memory
  // Reduce in powers of two
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Final stage: use warp-level primitives for the last 32 elements
  float final_sum = (threadIdx.x < 32) ? sdata[threadIdx.x] : 0.0f;
  // Unroll warp-level reduction using __shfl_down_sync
  for (int offset = 16; offset > 0; offset /= 2) {
    final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
  }

  // Thread 0 in the block writes the final result, adding bias if provided
  if (threadIdx.x == 0) {
    if (bias != nullptr) {
      final_sum += bias[oc];
    }
    output[out_idx] = final_sum;
  }
}

// C++ forward function wrapping the CUDA kernel
// This function sets up the kernel launch parameters and invokes the kernel.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  if (bias_opt.has_value()) {
    CHECK_INPUT(*bias_opt);
  }

  // Extract input dimensions (assumed shape: [batch, in_channels, in_d, in_h, in_w])
  int batch = x.size(0);
  int in_channels = x.size(1);
  int in_d = x.size(2);
  int in_h = x.size(3);
  int in_w = x.size(4);

  // Extract kernel dimensions (assumed shape: [in_channels, out_channels_per_group, k_d, k_h, k_w])
  int k_d = weight.size(2);
  int k_h = weight.size(3);
  int k_w = weight.size(4);

  // Strides and padding
  int s_d = stride[0];
  int s_h = stride[1];
  int s_w = stride[2];
  int p_d = padding[0];
  int p_h = padding[1];
  int p_w = padding[2];
  int op_d = output_padding[0];
  int op_h = output_padding[1];
  int op_w = output_padding[2];

  // Compute output dimensions for transposed convolution
  int out_d = (in_d - 1) * s_d - 2 * p_d + k_d + op_d;
  int out_h = (in_h - 1) * s_h - 2 * p_h + k_h + op_h;
  int out_w = (in_w - 1) * s_w - 2 * p_w + k_w + op_w;

  // Determine output channels
  int channels_per_group_out = weight.size(1);
  int out_channels = channels_per_group_out * groups;

  // Compute number of input channels per group
  int channels_per_group_in = in_channels / groups;

  // Allocate output tensor with shape: [batch, out_channels, out_d, out_h, out_w]
  auto output = torch::zeros({batch, out_channels, out_d, out_h, out_w}, x.options());

  // Get raw pointers for kernel launch
  const float* x_ptr = x.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const float* bias_ptr = bias_opt.has_value() ? (*bias_opt).data_ptr<float>() : nullptr;
  float* out_ptr = output.data_ptr<float>();

  // Total number of output elements
  int total_outputs = batch * out_channels * out_d * out_h * out_w;

  // Launch one block per output element
  // Choose block size, e.g., 256 threads per block
  int block_size = 256;
  int num_blocks = total_outputs;
  // Shared memory size (per block) in bytes
  size_t shared_mem_size = block_size * sizeof(float);

  transposed_conv3d_shared_reduction_kernel<<<num_blocks, block_size, shared_mem_size>>>(
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
      channels_per_group_out);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward with shared memory and warp-level reduction (CUDA)");
}
