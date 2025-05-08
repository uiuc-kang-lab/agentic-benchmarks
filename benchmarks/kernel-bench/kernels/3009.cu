#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Generic kernel for non-float types using __ldg for read-only loads
// (Note: __ldg works with double as well, but vectorization is applied for float)
template <typename scalar_t>
__global__ void tanh_ldg_generic_kernel(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          const int numel) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x;
  for (int i = tid; i < numel; i += stride) {
    // Use __ldg for read-only memory fetch
    output[i] = (std::is_same<scalar_t, float>::value) ? tanhf(__ldg(input + i)) : tanh(__ldg(input + i));
  }
}

// Specialized kernel for float using vectorized 128-bit (float4) loads and stores
// and __ldg() for read-only accesses
__global__ void tanh_ldg_kernel_float4(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         const int numel) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x;

  // Process full groups of 4 floats (128 bits per load/store)
  int num_vec = numel / 4;
  for (int i = tid; i < num_vec; i += stride) {
    // Load a float4 from aligned global memory using __ldg for read-only caching
    float4 in = __ldg(reinterpret_cast<const float4*>(input) + i);
    float4 out;
    out.x = tanhf(in.x);
    out.y = tanhf(in.y);
    out.z = tanhf(in.z);
    out.w = tanhf(in.w);
    reinterpret_cast<float4*>(output)[i] = out;
  }

  // Process any remaining elements that don't form a complete vector
  int rem_start = num_vec * 4;
  for (int i = tid; i < (numel - rem_start); i += stride) {
    output[rem_start + i] = tanhf(__ldg(input + rem_start + i));
  }
}

// Forward function exposed to Python
torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int numel = input.numel();
  const int threads = 256;

  if (input.scalar_type() == at::ScalarType::Float) {
    // Use vectorized kernel for float
    int num_vec = numel / 4;
    int blocks = (num_vec + threads - 1) / threads;
    if(blocks < 1) blocks = 1;
    tanh_ldg_kernel_float4<<<blocks, threads>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      numel
    );
  } else {
    // Use the generic kernel for other floating point types
    int blocks = (numel + threads - 1) / threads;
    if(blocks < 1) blocks = 1;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_ldg_generic_kernel", ([&] {
      tanh_ldg_generic_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        numel
      );
    }));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Tanh forward using __ldg for optimized 128-bit aligned loads (CUDA)");
}
