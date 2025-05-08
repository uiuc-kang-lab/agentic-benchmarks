#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

// Fallback scalar kernel using __ldg for read-only loads
template <typename scalar_t>
__global__ void hardtanh_kernel(const scalar_t* __restrict__ x,
                                 scalar_t* __restrict__ out,
                                 int64_t numel,
                                 scalar_t min_val,
                                 scalar_t max_val) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) {
    scalar_t val = __ldg(&x[i]);
    if (val < min_val) {
      val = min_val;
    } else if (val > max_val) {
      val = max_val;
    }
    out[i] = val;
  }
}

// Optimized kernel for float using 128-bit aligned vectorized loads (float4) 
__global__ void hardtanh_kernel_vec_float(const float* __restrict__ x,
                                           float* __restrict__ out,
                                           int64_t numel,
                                           float min_val,
                                           float max_val) {
    const int vec_size = 4; // 4 floats = 128 bits
    int vec_elems = numel / vec_size;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < vec_elems) {
         const float4* x_vec = reinterpret_cast<const float4*>(x);
         float4 v = __ldg(&x_vec[tid]);
         float4 result;
         result.x = (v.x < min_val ? min_val : (v.x > max_val ? max_val : v.x));
         result.y = (v.y < min_val ? min_val : (v.y > max_val ? max_val : v.y));
         result.z = (v.z < min_val ? min_val : (v.z > max_val ? max_val : v.z));
         result.w = (v.w < min_val ? min_val : (v.w > max_val ? max_val : v.w));
         float4* out_vec = reinterpret_cast<float4*>(out);
         out_vec[tid] = result;
    }
    // Tail processing for elements that don't form a complete vector
    int tail = numel % vec_size;
    int offset = vec_elems * vec_size;
    for (int i = tid; i < tail; i += blockDim.x * gridDim.x) {
         float v = __ldg(&x[offset + i]);
         out[offset + i] = (v < min_val ? min_val : (v > max_val ? max_val : v));
    }
}

// Optimized kernel for double using 128-bit aligned vectorized loads (double2)
__global__ void hardtanh_kernel_vec_double(const double* __restrict__ x,
                                            double* __restrict__ out,
                                            int64_t numel,
                                            double min_val,
                                            double max_val) {
    const int vec_size = 2; // 2 doubles = 128 bits
    int vec_elems = numel / vec_size;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < vec_elems) {
         const double2* x_vec = reinterpret_cast<const double2*>(x);
         double2 v = __ldg(&x_vec[tid]);
         double2 result;
         result.x = (v.x < min_val ? min_val : (v.x > max_val ? max_val : v.x));
         result.y = (v.y < min_val ? min_val : (v.y > max_val ? max_val : v.y));
         double2* out_vec = reinterpret_cast<double2*>(out);
         out_vec[tid] = result;
    }
    // Tail processing
    int tail = numel % vec_size;
    int offset = vec_elems * vec_size;
    for (int i = tid; i < tail; i += blockDim.x * gridDim.x) {
         double v = __ldg(&x[offset + i]);
         out[offset + i] = (v < min_val ? min_val : (v > max_val ? max_val : v));
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();
  const int threads = 1024;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
    if (std::is_same<scalar_t, float>::value) {
        int vec_elems = numel / 4; // number of 128-bit chunks
        int blocks = (vec_elems + threads - 1) / threads;
        hardtanh_kernel_vec_float<<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel,
            static_cast<scalar_t>(min_val),
            static_cast<scalar_t>(max_val)
        );
    } else if (std::is_same<scalar_t, double>::value) {
        int vec_elems = numel / 2; // number of 128-bit chunks for double
        int blocks = (vec_elems + threads - 1) / threads;
        hardtanh_kernel_vec_double<<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel,
            static_cast<scalar_t>(min_val),
            static_cast<scalar_t>(max_val)
        );
    } else {
        int blocks = (numel + threads - 1) / threads;
        hardtanh_kernel<scalar_t><<<blocks, threads>>>(
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

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (!x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CUDA tensor");
  }
  return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh activation (CUDA) with optimized, vectorized memory accesses");
}
