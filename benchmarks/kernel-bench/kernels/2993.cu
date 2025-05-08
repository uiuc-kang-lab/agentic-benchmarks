#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Generic kernel for non-float types (or fallback for vectorization)
template <typename scalar_t>
__global__ void tanh_kernel_no_sync_generic(const scalar_t* __restrict__ input,
                                              scalar_t* __restrict__ output,
                                              const int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;
  for (int i = tid; i < size; i += total_threads) {
    if constexpr (std::is_same<scalar_t, float>::value) {
      output[i] = tanhf(input[i]);
    } else {
      output[i] = tanh(input[i]);
    }
  }
}

// Specialized kernel using vectorized loads/stores for float type.
// Note: No __syncthreads() is used because each thread operates independently.
__global__ void tanh_kernel_no_sync_float4(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             const int num_elements) {
  int total_threads = blockDim.x * gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Process most elements using vectorized float4 load/store
  int vec_size = num_elements / 4;  // number of complete groups of 4
  for (int idx = tid; idx < vec_size; idx += total_threads) {
    float4 in = reinterpret_cast<const float4*>(input)[idx];
    float4 out;
    out.x = tanhf(in.x);
    out.y = tanhf(in.y);
    out.z = tanhf(in.z);
    out.w = tanhf(in.w);
    reinterpret_cast<float4*>(output)[idx] = out;
  }

  // Process any remaining elements that don't form a complete vector
  int rem_start = vec_size * 4;
  for (int i = rem_start + tid; i < num_elements; i += total_threads) {
    output[i] = tanhf(input[i]);
  }
}


// Forward function exposed to Python
torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int numel = input.numel();
  const int threads = 256;
  // Compute blocks so that we cover all elements in our stride loop
  const int blocks = (numel + (threads * 4) - 1) / (threads * 4);

  if (input.scalar_type() == at::ScalarType::Float) {
    // Use the specialized kernel for floats with vectorized accesses
    tanh_kernel_no_sync_float4<<<blocks, threads>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      numel
    );
  } else {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_no_sync_generic", ([&] {
      tanh_kernel_no_sync_generic<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        numel
      );
    }));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized Tanh forward without unnecessary synchronizations (CUDA)");
}
