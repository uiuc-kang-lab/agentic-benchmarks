#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

// Fallback scalar kernel using __ldg() for read-only loads
template <typename scalar_t>
__global__ void hardtanh_scalar_kernel(const scalar_t* __restrict__ x,
                                         scalar_t* __restrict__ out,
                                         int64_t numel,
                                         scalar_t min_val,
                                         scalar_t max_val) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) {
    scalar_t val = __ldg(x + i);
    out[i] = (val < min_val ? min_val : (val > max_val ? max_val : val));
  }
}

// Optimized kernel for float using 128-bit loads/stores via float4
__global__ void hardtanh_float4_aligned_kernel(const float* __restrict__ x,
                                                 float* __restrict__ out,
                                                 int64_t numel,
                                                 float min_val,
                                                 float max_val) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int vecCount = numel / 4;
  if (index < vecCount) {
    // Load 128 bits (4 floats) using __ldg for read-only access
    float4 in_val = __ldg(reinterpret_cast<const float4*>(x) + index);
    in_val.x = fminf(fmaxf(in_val.x, min_val), max_val);
    in_val.y = fminf(fmaxf(in_val.y, min_val), max_val);
    in_val.z = fminf(fmaxf(in_val.z, min_val), max_val);
    in_val.w = fminf(fmaxf(in_val.w, min_val), max_val);
    // Store aligned 128-bit
    reinterpret_cast<float4*>(out)[index] = in_val;
  }
}

// Optimized kernel for double using 128-bit loads/stores via double2
__global__ void hardtanh_double2_aligned_kernel(const double* __restrict__ x,
                                                  double* __restrict__ out,
                                                  int64_t numel,
                                                  double min_val,
                                                  double max_val) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int vecCount = numel / 2;
  if (index < vecCount) {
    // Load 128 bits (2 doubles) using __ldg
    double2 in_val = __ldg(reinterpret_cast<const double2*>(x) + index);
    in_val.x = (in_val.x < min_val ? min_val : (in_val.x > max_val ? max_val : in_val.x));
    in_val.y = (in_val.y < min_val ? min_val : (in_val.y > max_val ? max_val : in_val.y));
    reinterpret_cast<double2*>(out)[index] = in_val;
  }
}

at::Tensor forward_cuda(const at::Tensor& x, double min_val, double max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();
  // Tune block size based on compute capability
const int threads = 128; // Reduced thread count for better occupancy balance
  
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    if (std::is_same<scalar_t, float>::value) {
      // Use vectorized loads/stores on 128-bit boundaries when possible
      if (numel % 4 == 0) {
        int vecCount = numel / 4;
        int blocks = (vecCount + threads - 1) / threads;
        hardtanh_float4_aligned_kernel<<<blocks, threads>>>(
          x.data_ptr<float>(),
          out.data_ptr<float>(),
          numel,
          static_cast<float>(min_val),
          static_cast<float>(max_val)
        );
      } else {
        int blocks = (numel + threads - 1) / threads;
        hardtanh_scalar_kernel<scalar_t><<<blocks, threads>>>(
          x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),
          numel,
          static_cast<scalar_t>(min_val),
          static_cast<scalar_t>(max_val)
        );
      }
    } else if (std::is_same<scalar_t, double>::value) {
      // For double, 2 doubles equal 128 bits
      if (numel % 2 == 0) {
        int vecCount = numel / 2;
        int blocks = (vecCount + threads - 1) / threads;
        hardtanh_double2_aligned_kernel<<<blocks, threads>>>(
          x.data_ptr<double>(),
          out.data_ptr<double>(),
          numel,
          static_cast<double>(min_val),
          static_cast<double>(max_val)
        );
      } else {
        int blocks = (numel + threads - 1) / threads;
        hardtanh_scalar_kernel<scalar_t><<<blocks, threads>>>(
          x.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),
          numel,
          static_cast<scalar_t>(min_val),
          static_cast<scalar_t>(max_val)
        );
      }
    } else {
      int blocks = (numel + threads - 1) / threads;
      hardtanh_scalar_kernel<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        numel,
        static_cast<scalar_t>(min_val),
        static_cast<scalar_t>(max_val)
      );
    }
  }));
  
  return out;
}

at::Tensor forward(const at::Tensor& x, double min_val, double max_val) {
  if (!x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CUDA tensor");
  }
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh activation (CUDA) optimized with 128-bit alignment");
}
