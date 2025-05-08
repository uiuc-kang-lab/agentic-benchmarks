#include <torch/extension.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

__global__
void conv_transpose_3d_kernel(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              float* __restrict__ output,
                              int64_t x_dim0, int64_t x_dim1, int64_t x_dim2, 
                              int64_t x_dim3, int64_t x_dim4, 
                              int64_t weight_dim0, int64_t weight_dim1,
                              int64_t weight_dim2, int64_t weight_dim3, int64_t weight_dim4,
                              int64_t stride0, int64_t stride1, int64_t stride2,
                              int64_t padding0, int64_t padding1, int64_t padding2,
                              int64_t groups) {
  int64_t batch = blockIdx.x;
  int64_t in_channel_group = blockIdx.y * blockDim.y + threadIdx.y; // Group-wise thread distribution
  int64_t channel_within_group = in_channel_group % (weight_dim0 / groups);
  
  float sum = 0.0f;
  int64_t kernel_volume = weight_dim1 * weight_dim2 * weight_dim3;
  
  for (int64_t k = 0; k < kernel_volume; ++k) {
    int64_t ki = k / (weight_dim2 * weight_dim3);
    int64_t kj = (k / weight_dim3) % weight_dim2;
    int64_t kk = k % weight_dim3;
    
    int64_t x0 = (blockIdx.z * stride0 - padding0) + ki;
    int64_t x1 = (blockIdx.y * stride1 - padding1) + kj;
    int64_t x2 = threadIdx.x * stride2 - padding2 + kk;
    
    if (x0 >= 0 && x0 < x_dim1 && x1 >= 0 && x1 < x_dim2 && x2 >= 0 && x2 < x_dim3) {
      int64_t weight_index = channel_within_group * kernel_volume * weight_dim4
                            + ki * weight_dim2 * weight_dim3 * weight_dim4
                            + kj * weight_dim3 * weight_dim4
                            + kk * weight_dim4
                            + threadIdx.x;
      int64_t x_index = batch * x_dim1 * x_dim2 * x_dim3 * x_dim4
                        + in_channel_group * x_dim2 * x_dim3 * x_dim4
                        + x0 * x_dim3 * x_dim4
                        + x1 * x_dim4
                        + x2;
      sum += x[x_index] * weight[weight_index];
    }
  }

  // Only use atomic operations at output write to handle potential race conditions
  atomicAdd(&output[batch * gridDim.y * gridDim.z * blockDim.x + blockIdx.y * gridDim.z * blockDim.x + blockIdx.z * blockDim.x + threadIdx.x], sum);
}

// Function definition matching the expected parameters
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  if (bias.has_value()) {
    CHECK_INPUT(*bias);
  }

  auto output = torch::zeros({/* Output size calculations */}, x.options());

  /* Define block dimensions: use block.x for output channels and block.y for input channel tiles */
// We assume that weight.size(4) = out_channels and we choose a block of out_channels x 8 threads
// Adjust these values as needed for performance
const int block_oc = weight.size(4);
const int block_ic = 8;
dim3 threads(block_oc, block_ic, 1);
  dim3 blocks(/* appropriate grid size based on output size */);
  conv_transpose_3d_kernel<<<blocks, threads>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      output.data_ptr<float>(),
      x.size(0), x.size(1), x.size(2), x.size(3), x.size(4),
      weight.size(0), weight.size(1), weight.size(2), weight.size(3), weight.size(4),
      stride[0], stride[1], stride[2],
      padding[0], padding[1], padding[2],
      groups);

  if (bias.has_value()) {
    output += *bias;
  }

  return output;
}

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized Transposed Conv3D forward (CUDA)");
}