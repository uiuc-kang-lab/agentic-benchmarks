#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Tile dimensions for output
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// This kernel uses tiling with shared memory to load a contiguous region of the input,
// ensuring that global memory accesses are coalesced. Each block processes a tile of the output
// corresponding to one (batch, channel) plane. The shared memory tile is sized to cover the pooling
// windows for the entire output tile, so that threads in a warp load consecutive memory locations.

template <typename scalar_t>
__global__ void maxpool2d_tiled_coalesced_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

  // Determine which (batch, channel) plane this block will process
  int bc = blockIdx.z;
  int b = bc / channels;
  int c = bc % channels;

  // Calculate the starting indices of the output tile for this block
  int out_tile_start_x = blockIdx.x * TILE_WIDTH;
  int out_tile_start_y = blockIdx.y * TILE_HEIGHT;

  // Global output coordinate for each thread within the tile
  int tx = threadIdx.x; // [0, TILE_WIDTH)
  int ty = threadIdx.y; // [0, TILE_HEIGHT)
  int out_x = out_tile_start_x + tx;
  int out_y = out_tile_start_y + ty;

  // Determine shared memory dimensions.
  // For each output element, the pooling window starts at (output_coord * stride - padding).
  // Thus, the shared memory tile must cover the entire tile of outputs plus the extra border for the pooling window.
  int smem_width = TILE_WIDTH * stride + (kernel_size - 1) * dilation;
  int smem_height = TILE_HEIGHT * stride + (kernel_size - 1) * dilation;

  // Compute the origin (top-left corner) of the shared memory tile in the global input
  int in_tile_start_x = out_tile_start_x * stride - padding;
  int in_tile_start_y = out_tile_start_y * stride - padding;

  // Allocate shared memory (dynamically allocated as char array, then reinterpreted as scalar_t)
  extern __shared__ char shared_data[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_data);

  // Total number of elements in shared memory
  int smem_size = smem_width * smem_height;
  int num_threads = TILE_WIDTH * TILE_HEIGHT;
  int tid = ty * TILE_WIDTH + tx;

  // Each thread loads multiple elements from global memory into shared memory in a coalesced manner
  for (int i = tid; i < smem_size; i += num_threads) {
      int r = i / smem_width;
      int col = i % smem_width;
      // Corresponding global input coordinates
      int in_y = in_tile_start_y + r;
      int in_x = in_tile_start_x + col;
      // Compute the base index for the (b, c) plane
      int input_base = b * channels * input_height * input_width + c * input_height * input_width;
      if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
          sdata[i] = input[input_base + in_y * input_width + in_x];
      } else {
          sdata[i] = -std::numeric_limits<scalar_t>::infinity();
      }
  }
  __syncthreads();

  // Each thread computes one output element if within the valid output range
  if (out_x < output_width && out_y < output_height) {
      // The top-left coordinate of the pooling window in shared memory
      // For an output at (out_y, out_x), the corresponding shared memory coordinate is:
      // local_x = (out_x - out_tile_start_x) * stride
      // local_y = (out_y - out_tile_start_y) * stride
      int local_x = tx * stride;
      int local_y = ty * stride;
      scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

      // Iterate over the pooling window
      for (int kh = 0; kh < kernel_size; ++kh) {
          for (int kw = 0; kw < kernel_size; ++kw) {
              int lx = local_x + kw * dilation;
              int ly = local_y + kh * dilation;
              int smem_index = ly * smem_width + lx;
              scalar_t val = sdata[smem_index];
              if (val > max_val) {
                  max_val = val;
              }
          }
      }
      int output_base = b * channels * output_height * output_width + c * output_height * output_width;
      int output_index = output_base + out_y * output_width + out_x;
      output[output_index] = max_val;
  }
}


// Host function to launch the tiled, coalesced max pooling kernel
torch::Tensor maxpool2d_tiled_coalesced_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

  int batch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  // Calculate output dimensions based on the pooling parameters
  int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
  int output_width  = ((input_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

  auto output = torch::empty({batch, channels, output_height, output_width}, input.options());

  // Grid dimensions: x and y for output spatial tiles, z for (batch * channels)
  dim3 block(TILE_WIDTH, TILE_HEIGHT);
  dim3 grid((output_width + TILE_WIDTH - 1) / TILE_WIDTH,
            (output_height + TILE_HEIGHT - 1) / TILE_HEIGHT,
            batch * channels);

  // Shared memory size: space for the input tile for this block
  int smem_width = TILE_WIDTH * stride + (kernel_size - 1) * dilation;
  int smem_height = TILE_HEIGHT * stride + (kernel_size - 1) * dilation;
  size_t shared_mem_bytes = smem_width * smem_height * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool2d_tiled_coalesced_cuda_forward", ([&] {
    maxpool2d_tiled_coalesced_kernel<scalar_t><<<grid, block, shared_mem_bytes>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        dilation);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &maxpool2d_tiled_coalesced_cuda_forward, "Tiled Coalesced MaxPool2D forward (CUDA)");
}
