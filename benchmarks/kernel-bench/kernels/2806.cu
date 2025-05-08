#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// The kernel uses __ldg() for read-only global memory loads and vectorized loads/stores
// to ensure 128-bit aligned memory accesses. This improves memory bandwidth and reduces latency.

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Specialize for float using float4 vectorization (128-bit loads)
  if constexpr (std::is_same<scalar_t, float>::value) {
    int vec_size = size / 4;  // number of complete float4 groups
    float4* out_vec = reinterpret_cast<float4*>(output);
    const float4* in_vec = reinterpret_cast<const float4*>(input);
    
    // Process vectorized portion
    for (int i = idx; i < vec_size; i += stride) {
      // Use __ldg() for read-only load
      float4 in_val = __ldg(&in_vec[i]);
      float y0 = 1.0f / (1.0f + __expf(-in_val.x));
      float y1 = 1.0f / (1.0f + __expf(-in_val.y));
      float y2 = 1.0f / (1.0f + __expf(-in_val.z));
      float y3 = 1.0f / (1.0f + __expf(-in_val.w));
      float4 out_val = make_float4(y0, y1, y2, y3);
      out_vec[i] = out_val;
    }
    
    // Process any remaining elements
    int leftover = size - vec_size * 4;
    int start = vec_size * 4;
    for (int i = idx; i < leftover; i += stride) {
      int index = start + i;
      float in_val = __ldg(&input[index]);
      output[index] = 1.0f / (1.0f + __expf(-in_val));
    }
  } 
  // Specialize for double using double2 vectorization
  else if constexpr (std::is_same<scalar_t, double>::value) {
    int vec_size = size / 2;  // number of complete double2 groups (2*64 bits = 128 bits)
    double2* out_vec = reinterpret_cast<double2*>(output);
    const double2* in_vec = reinterpret_cast<const double2*>(input);

    // Process vectorized portion
    for (int i = idx; i < vec_size; i += stride) {
      // __ldg() used for read-only load
      double2 in_val = __ldg(&in_vec[i]);
      double y0 = 1.0 / (1.0 + exp(-in_val.x));
      double y1 = 1.0 / (1.0 + exp(-in_val.y));
      double2 out_val;
      out_val.x = y0;
      out_val.y = y1;
      out_vec[i] = out_val;
    }

    // Process remaining element if size is odd
    int leftover = size - vec_size * 2;
    int start = vec_size * 2;
    for (int i = idx; i < leftover; i += stride) {
      int index = start + i;
      double in_val = __ldg(&input[index]);
      output[index] = 1.0 / (1.0 + exp(-in_val));
    }
  }
}


torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();
  
  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;
  
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", ([&] {
    const auto* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    sigmoid_kernel<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
  }));
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Sigmoid forward (CUDA) with __ldg() and 128-bit aligned accesses");
}
