#include <torch/extension.h>

// Kernel that maps threads in a 3D grid corresponding to the output tensor's dimensions.
// The grid's z-dimension covers the batch and channel dimensions (N and C_out), while the x and y dimensions cover the spatial dimensions (W_out and H_out).
__global__ void add_bias_kernel_3d(
    float* output,          // pointer to the output tensor
    const float* bias,      // pointer to the bias tensor
    int N,                  // batch size
    int C_out,              // number of output channels
    int H_out,              // output height
    int W_out) {            // output width
  // Compute the batch and channel indices from the grid's z-dimension
  int idx = blockIdx.z;
  int n = idx / C_out;
  int c = idx % C_out;

  // Compute the spatial indices using 2D block indexing
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;

  // Check bounds for spatial dimensions
  if (h < H_out && w < W_out) {
    // Compute the linear index in the output tensor assuming NCHW layout
    int offset = ((n * C_out + c) * H_out + h) * W_out + w;
    output[offset] += bias[c];
  }
}

// Forward function definition
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

  // Ensure inputs are on CUDA and contiguous
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
  TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
  TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

  if (bias.has_value()) {
    TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
    TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
  }

  // Use the built-in conv_transpose2d function for the main computation
  auto output = at::conv_transpose2d(
      x,
      weight,
      bias,
      {stride, stride},                 // stride
      {padding, padding},               // padding
      {output_padding, output_padding}, // output_padding
      groups
  );

  // If bias is provided, add it using our 3D-mapped kernel for efficient thread mapping
  if (bias.has_value()) {
    int N = x.size(0);
    int C_out = weight.size(1);
    int H_out = output.size(2);
    int W_out = output.size(3);

    // Define 2D block size for spatial dimensions
    dim3 block(16, 16);
    // Grid dimensions: x dimension for W, y dimension for H, and z dimension for (N * C_out)
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C_out
    );

    add_bias_kernel_3d<<<grid, block>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        N,
        C_out,
        H_out,
        W_out
    );
    cudaDeviceSynchronize();
  }

  return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA) - 3D mapped bias addition");
}
