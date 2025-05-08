#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

inline int next_power_of_two(int x) {
    return 1 << (32 - __builtin_clz(x - 1));
}

// Warp reduce max
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Block reduce max
template <typename scalar_t>
__device__ scalar_t block_reduce_max(scalar_t val) {
  static __shared__ scalar_t shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warp_reduce_max(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -std::numeric_limits<scalar_t>::infinity();
  if (wid == 0) val = warp_reduce_max(val);

  return val;
}

// Warp reduce sum
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Block reduce sum
template <typename scalar_t>
__device__ scalar_t block_reduce_sum(scalar_t val) {
  static __shared__ scalar_t shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warp_reduce_sum(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
  if (wid == 0) val = warp_reduce_sum(val);

  return val;
}


template <typename scalar_t>
__global__ void log_softmax_no_divergence_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    int batch_idx = blockIdx.x;

    const scalar_t* input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    // First pass: find max value
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
        thread_max = max(thread_max, input_row[idx]);
    }
    // Reduce max across block
    scalar_t max_val = block_reduce_max(thread_max);
    __syncthreads();  // Ensure all threads have the same max_val

    // Second pass: compute exp sum
    scalar_t thread_sum = 0;
    for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
        scalar_t val = exp(input_row[idx] - max_val);
        output_row[idx] = val;  // Store intermediate results
        thread_sum += val;
    }
    // Reduce sum across block
    scalar_t sum = block_reduce_sum(thread_sum);
    __syncthreads();  // Ensure all threads have the same sum

    // Avoid log(0)
    sum = max(sum, std::numeric_limits<scalar_t>::min());
    scalar_t log_sum = log(sum);

    // Final pass: compute log softmax
    for (int idx = threadIdx.x; idx < dim_size; idx += blockDim.x) {
        output_row[idx] = input_row[idx] - max_val - log_sum;
    }
}

torch::Tensor log_softmax_no_divergence_cuda_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat64,
        "input must be float32 or float64");

    int64_t ndim = input.dim();
    TORCH_CHECK(dim >= -ndim && dim < ndim, "dim out of range");

    dim = dim >= 0 ? dim : dim + ndim;

    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            permute_dims.push_back(i);
        }
    }
    permute_dims.push_back(dim);

    input = input.permute(permute_dims).contiguous();
    int64_t batch_size = input.numel() / input.size(-1);
    int64_t dim_size = input.size(-1);

    auto output = torch::empty_like(input);

    int threads = next_power_of_two(dim_size);
    threads = threads < 1024 ? threads : 1024;

    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_no_divergence_forward_cuda", ([&] {
        log_softmax_no_divergence_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size);
    }));

    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    output = output.permute(inverse_permute_dims);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &log_softmax_no_divergence_cuda_forward, "LogSoftmax no divergence forward (CUDA)");
}