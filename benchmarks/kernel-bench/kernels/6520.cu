#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// New optimized kernel using vectorized loads for float types when inner_size is divisible by 4
__global__ void mean_reduce_kernel_vec(
    const float* input,
    float* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {
  // inner_size is assumed to be divisible by 4
  int inner_vec = inner_size / 4;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= outer_size * inner_vec) return;
  int outer_idx = tid / inner_vec;
  int inner_idx = (tid % inner_vec) * 4;

  // Compute offset in the vectorized view
  int input_offset_vec = outer_idx * dim_size * inner_vec;
  const float4* input_vec = reinterpret_cast<const float4*>(input);

  float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int i = 0; i < dim_size; i++) {
    int offset = input_offset_vec + i * inner_vec + (inner_idx / 4);
    float4 val = input_vec[offset];
    accum.x += val.x;
    accum.y += val.y;
    accum.z += val.z;
    accum.w += val.w;
  }
  accum.x /= dim_size;
  accum.y /= dim_size;
  accum.z /= dim_size;
  accum.w /= dim_size;

  int output_offset = outer_idx * inner_size + inner_idx;
  output[output_offset + 0] = accum.x;
  output[output_offset + 1] = accum.y;
  output[output_offset + 2] = accum.z;
  output[output_offset + 3] = accum.w;
}


template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t outer_size,
    int64_t dim_size, 
    int64_t inner_size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= outer_size * inner_size) return;
    
    const int outer_idx = tid / inner_size;
    const int inner_idx = tid % inner_size;
    const int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    scalar_t sum = 0;
    #pragma unroll
    for (int i = 0; i < dim_size; i++) {
        sum += input[input_offset + i * inner_size];
    }
    
    output[tid] = sum / dim_size;
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA)");
}
