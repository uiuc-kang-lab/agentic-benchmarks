#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void tanh_grid_stride_kernel(const T* __restrict__ input,
                                       T* __restrict__ output,
                                       int numel) {
  const int stride = blockDim.x * gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Process all elements with grid-stride pattern
  while (tid < numel) {
    output[tid] = tanh(input[tid]);
    tid += stride;
  }
}

// Specialized optimized version for float4 vectorization
__global__ void tanh_grid_stride_vectorized(float* __restrict__ input,
                                            float* __restrict__ output,
                                            int numel) {
  const int vec_size = 4;
  const int num_vec = numel / vec_size;
  const int total_vecs = num_vec + ((numel % vec_size) ? 1 : 0);
  const int stride = blockDim.x * gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const int rem = numel % vec_size;
  const int total_iters = num_vec + (rem ? 1 : 0);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Process both full vectors and remaining elements in a single loop
  while (i < total_iters) {
    if (i < num_vec) {
      float4 vec_in = __ldg(reinterpret_cast<const float4*>(input) + i);
      float4 vec_out;
      vec_out.x = tanhf(vec_in.x);
      vec_out.y = tanhf(vec_in.y);
      vec_out.z = tanhf(vec_in.z);
      vec_out.w = tanhf(vec_in.w);
      reinterpret_cast<float4*>(output)[i] = vec_out;
    } else {
      int j = num_vec * vec_size + (i - num_vec);
      if (j < numel) {
        output[j] = tanhf(__ldg(input + j));
      }
    }
    i += stride;
  }
}

torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  const int numel = input.numel();
  const int threads = 512;
  
  if (input.scalar_type() == at::kFloat) {
    const int vec_numel = numel / 4;
    int blocks = (vec_numel + threads - 1) / threads;
    blocks = max(blocks, 1);
    tanh_grid_stride_vectorized<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
  } else {
    int blocks = (numel + threads - 1) / threads;
    blocks = max(blocks, 1);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_grid_stride", [&] {
      tanh_grid_stride_kernel<scalar_t><<<blocks, threads>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          numel
      );
    });
  }
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Grid-stride vectorized Tanh forward (CUDA)");
}