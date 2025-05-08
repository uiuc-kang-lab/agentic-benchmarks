#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Generic kernel using stride loops for all floating point types
template <typename scalar_t>
__global__ void tanh_stride_kernel(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     const int numel) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; // Thread ID
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < numel; i += stride) {
    if constexpr (std::is_same<scalar_t, float>::value) {
      output[i] = tanhf(input[i]);
    } else {
      output[i] = tanh(input[i]);
    }
  }
}

// Specialized kernel for float using vectorized float4 loads/stores with stride loops
__global__ void tanh_stride_kernel_float4(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            const int numel) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; // Thread ID
  int stride = blockDim.x * gridDim.x;

  // Process complete groups of 4 floats
  int numVec = numel / 4;  // number of complete float4 groups
  for (int i = tid; i < numVec; i += stride) {
    float4 in = reinterpret_cast<const float4*>(input)[i];
    float4 out;
    out.x = tanhf(in.x);
    out.y = tanhf(in.y);
    out.z = tanhf(in.z);
    out.w = tanhf(in.w);
    reinterpret_cast<float4*>(output)[i] = out;
  }

  // Process any remaining elements that don't fit in a complete vector
  int start = numVec * 4;
  for (int i = start + tid; i < numel; i += stride) {
    output[i] = tanhf(input[i]);
  }
}


// Forward function exposed to Python
torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  int numel = input.numel();
  const int threads = 256;

  if (input.scalar_type() == at::ScalarType::Float) {
    // Calculate blocks based on the number of complete float4 groups
    int numVec = numel / 4;
    int blocks = (numVec + threads - 1) / threads;
    blocks = (blocks > 0) ? blocks : 1;
    tanh_stride_kernel_float4<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);
  } else {
    // For other floating point types, use the generic kernel
    int blocks = (numel + threads - 1) / threads;
    blocks = (blocks > 0) ? blocks : 1;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_stride_kernel", ([&] {
      tanh_stride_kernel<scalar_t><<<blocks, threads>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          numel);
    }));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Tanh forward with stride loops (CUDA)");
}
