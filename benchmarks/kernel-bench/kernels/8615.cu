#include <torch/extension.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

// Constants, adjust size as per requirement
__constant__ float const_weight[1024];

// Kernel function using constant memory
__global__ void conv_transposed3d_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    const int input_dim[3], 
    const int output_dim[3], 
    const int weight_dim[3], 
    const int stride[3], 
    const int padding[3], 
    const int groups) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if (idx < output_dim[0] * output_dim[1] * output_dim[2]) {
    // Calculate 3D indices from linear thread index
    int z = idx / (output_dim[1] * output_dim[2]);
    int y = (idx / output_dim[2]) % output_dim[1];
    int x = idx % output_dim[2];

    // Perform convolution operation
    float val = 0.0f;
    for (int c = 0; c < weight_dim[0]; ++c) {
      for (int dz = 0; dz < weight_dim[1]; ++dz) {
        for (int dy = 0; dy < weight_dim[2]; ++dy) {
          for (int dx = 0; dx < weight_dim[3]; ++dx) {
            int in_z = z * stride[0] - padding[0] + dz;
            int in_y = y * stride[1] - padding[1] + dy;
            int in_x = x * stride[2] - padding[2] + dx;
            if (in_z >= 0 && in_z < input_dim[0] && in_y >= 0 && in_y < input_dim[1] && in_x >= 0 && in_x < input_dim[2]) {
              val += input[(in_z * input_dim[1] + in_y) * input_dim[2] + in_x] * const_weight[((c * weight_dim[1] + dz) * weight_dim[2] + dy) * weight_dim[3] + dx];
            }
          }
        }
      }
    }
    output[idx] = val;
  }
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

  // Copy weight to constant memory
  cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float));

  // Output tensor
  auto output = at::empty({/* appropriate sizes based on x, weight, etc.*/}, x.options());
  
  // Call the kernel (adjust block and grid sizes as needed)
  int threads = 256;
  int blocks = (output.numel() + threads - 1) / threads;
  
  conv_transposed3d_kernel<<<blocks, threads>>>(
    x.data_ptr<float>(),
    output.data_ptr<float>(),
    {x.size(0), x.size(1), x.size(2)},
    {output.size(0), output.size(1), output.size(2)},
    {weight.size(0), weight.size(1), weight.size(2), weight.size(3)},
    {stride[0], stride[1], stride[2]},
    {padding[0], padding[1], padding[2]},
    groups
  );

  return output;
}

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward (CUDA)");
}