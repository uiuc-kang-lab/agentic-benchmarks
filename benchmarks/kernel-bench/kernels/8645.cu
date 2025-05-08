#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Macros to verify tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// Optimized CUDA kernel using 3D thread block and grid indexing
// to efficiently map the 5D output [batch, out_channels, out_d, out_h, out_w]
// to the GPU threads.
__global__ void optimized_thread_block_indexing_kernel(
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
    int channels_per_group_out,
    int n_tile_d  // number of tiles along the depth dimension
    ) {
  // Tile sizes are defined by block dimensions
  const int tile_w = blockDim.x;  // e.g., 8
  const int tile_h = blockDim.y;  // e.g., 8
  const int tile_d = blockDim.z;  // e.g., 4

  // Map grid indices to output spatial coordinates:
  int out_w_idx = blockIdx.x * tile_w + threadIdx.x;
  int out_h_idx = blockIdx.y * tile_h + threadIdx.y;

  // For the depth and combined batch/channel dimension, we use gridIdx.z.
  // gridDim.z is set to: batch * out_channels * n_tile_d, where n_tile_d = ceil(out_d/tile_d)
  int tile_index = blockIdx.z % n_tile_d;   // Tile index along depth
  int bc_index   = blockIdx.z / n_tile_d;       // Combined index for batch and out_channel
  int out_d_idx = tile_index * tile_d + threadIdx.z;

  // Decode combined index into batch and channel indices
  int n = bc_index / out_channels;
  int oc = bc_index % out_channels;

  // Check if computed indices are within output bounds
  if (out_w_idx >= out_w || out_h_idx >= out_h || out_d_idx >= out_d)
      return;
      
  // Initialize accumulator with bias if provided
  float sum = (bias != nullptr) ? bias[oc] : 0.0f;
  
  // Determine group and intra-group channel index
  int group = oc / channels_per_group_out;
  int oc_in_group = oc % channels_per_group_out;

  // Compute base coordinates with padding
  int d_base = out_d_idx + p_d;
  int h_base = out_h_idx + p_h;
  int w_base = out_w_idx + p_w;

  // Loop over the kernel dimensions
  for (int kd = 0; kd < k_d; kd++) {
      int tmp_d = d_base - kd;
      if (tmp_d % s_d != 0) continue;
      int in_d_idx = tmp_d / s_d;
      if (in_d_idx < 0 || in_d_idx >= in_d) continue;
      
      for (int kh = 0; kh < k_h; kh++) {
          int tmp_h = h_base - kh;
          if (tmp_h % s_h != 0) continue;
          int in_h_idx = tmp_h / s_h;
          if (in_h_idx < 0 || in_h_idx >= in_h) continue;
          
          for (int kw = 0; kw < k_w; kw++) {
              int tmp_w = w_base - kw;
              if (tmp_w % s_w != 0) continue;
              int in_w_idx = tmp_w / s_w;
              if (in_w_idx < 0 || in_w_idx >= in_w) continue;
              
              // Loop over the input channels for this group
              for (int ic = 0; ic < channels_per_group_in; ic++) {
                  int in_channel = group * channels_per_group_in + ic;
                  int input_idx = n * (in_channels * in_d * in_h * in_w) +
                                  in_channel * (in_d * in_h * in_w) +
                                  in_d_idx * (in_h * in_w) +
                                  in_h_idx * in_w +
                                  in_w_idx;
                  
                  int weight_idx = in_channel * (channels_per_group_out * k_d * k_h * k_w) +
                                   oc_in_group * (k_d * k_h * k_w) +
                                   kd * (k_h * k_w) +
                                   kh * k_w +
                                   kw;
                  sum += input[input_idx] * weight[weight_idx];
              }
          }
      }
  }
  
  // Write the computed result to the output tensor
  int output_idx = n * (out_channels * out_d * out_h * out_w) +
                   oc * (out_d * out_h * out_w) +
                   out_d_idx * (out_h * out_w) +
                   out_h_idx * out_w +
                   out_w_idx;
  output[output_idx] = sum;
}


// C++ forward function: sets up grid dimensions using 3D mapping
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
    
    // Input dimensions: [batch, in_channels, in_d, in_h, in_w]
    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_d = x.size(2);
    int in_h = x.size(3);
    int in_w = x.size(4);
    
    // Weight dimensions: [in_channels, out_channels_per_group, k_d, k_h, k_w]
    int k_d = weight.size(2);
    int k_h = weight.size(3);
    int k_w = weight.size(4);
    
    // Stride, Padding, and Output Padding
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
    
    int channels_per_group_out = weight.size(1);
    int out_channels = channels_per_group_out * groups;
    int channels_per_group_in = in_channels / groups;
    
    // Allocate output tensor
    auto output = torch::zeros({batch, out_channels, out_d, out_h, out_w}, x.options());
    
    // Define tile sizes for 3D thread block
    const int tile_w = 8;
    const int tile_h = 8;
    const int tile_d = 4;

    // Setup grid dimensions:
    // Grid.x covers the width dimension
    int grid_x = (out_w + tile_w - 1) / tile_w;
    // Grid.y covers the height dimension
    int grid_y = (out_h + tile_h - 1) / tile_h;
    // For depth, we tile the depth dimension
    int n_tile_d = (out_d + tile_d - 1) / tile_d;
    // Grid.z is set to cover (batch * out_channels * n_tile_d)
    int grid_z = batch * out_channels * n_tile_d;

    dim3 block(tile_w, tile_h, tile_d);
    dim3 grid(grid_x, grid_y, grid_z);
    
    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = (bias_opt.has_value()) ? (*bias_opt).data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();
    
    // Launch kernel
    optimized_thread_block_indexing_kernel<<<grid, block>>>(
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
      channels_per_group_out,
      n_tile_d
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Transposed Conv3D with 3D Thread Block and Grid Indexing (CUDA)");
}
