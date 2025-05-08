#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

// Kernel using a 2D grid for block indexing to improve thread mapping.
template <typename scalar_t>
__global__ void hardtanh_kernel_2d(const scalar_t* __restrict__ x,
                                    scalar_t* __restrict__ out,
                                    int64_t numel,
                                    scalar_t min_val,
                                    scalar_t max_val) {
  // Compute a linear block index from a 2D grid
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int64_t idx = blockId * blockDim.x + threadIdx.x;
  // Total number of threads in the grid
  int64_t stride = gridDim.x * gridDim.y * blockDim.x;
  for (; idx < numel; idx += stride) {
    scalar_t val = x[idx];
    if (val < min_val) {
      val = min_val;
    } else if (val > max_val) {
      val = max_val;
    }
    out[idx] = val;
  }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  const int threads = 256;
  int numBlocks = (numel + threads - 1) / threads;
  // Configure a 2D grid to balance thread block mapping
  int grid_x = static_cast<int>(ceil(sqrt(static_cast<double>(numBlocks))));
  int grid_y = (numBlocks + grid_x - 1) / grid_x;
  dim3 blocks(grid_x, grid_y);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda_2d", ([&] {
    hardtanh_kernel_2d<scalar_t><<<blocks, threads>>>(
      x.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      numel,
      static_cast<scalar_t>(min_val),
      static_cast<scalar_t>(max_val)
    );
  }));

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CUDA tensor");
  }
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "2D Grid HardTanh activation (CUDA)");
}
