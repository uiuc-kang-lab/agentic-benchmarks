#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <cooperative_groups.h>
#include <math.h>

// This kernel distributes the LayerNorm workload evenly across threads and blocks by
// splitting each instance (row) into segments processed by multiple blocks. Using
// cooperative groups for grid-level synchronization, each block computes partial sums
// for its assigned segment, atomically adds its results to global accumulators, and then
// a designated thread per instance computes the final mean and inverse standard deviation.
// Finally, each block normalizes its segment of the instance using the computed values. This
// design avoids underutilization when the number of instances is small, ensuring balanced
// workload distribution across the GPU.


namespace cg = cooperative_groups;

// The kernel uses a cooperative groups grid sync and is launched with grid cooperative support.
// It assumes that total number of blocks = outer_size * blocks_per_inst, where each instance
// (of size normalized_size) is split into 'blocks_per_inst' segments.

template <typename scalar_t>
__global__ void layernorm_distributed_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int outer_size,
    const int blocks_per_inst,
    typename at::acc_type<scalar_t, true>* __restrict__ g_sum,
    typename at::acc_type<scalar_t, true>* __restrict__ g_sum_sq,
    typename at::acc_type<scalar_t, true>* __restrict__ g_mean,
    typename at::acc_type<scalar_t, true>* __restrict__ g_inv_std) {

  cg::grid_group grid = cg::this_grid();

  // Determine which instance this block is processing
  int global_block_idx = blockIdx.x; // total block index
  int instance_idx = global_block_idx % outer_size;
  int block_in_inst = global_block_idx / outer_size;

  // Calculate segment boundaries for this block
  int seg_length = (normalized_size + blocks_per_inst - 1) / blocks_per_inst;
  int seg_start = block_in_inst * seg_length;
  int seg_end = seg_start + seg_length;
  if (seg_end > normalized_size) seg_end = normalized_size;

  // Pointers to the data for this instance
  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Phase 1: Each block computes partial sums over its segment
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = seg_start + threadIdx.x; i < seg_end; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    local_sum += val;
    local_sum_sq += val * val;
  }

  // Use dynamic shared memory for block-level reduction
  extern __shared__ accscalar_t smem[];
  accscalar_t* shm_sum = smem;
  accscalar_t* shm_sum_sq = smem + blockDim.x;
  int tid = threadIdx.x;
  shm_sum[tid] = local_sum;
  shm_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shm_sum[tid] += shm_sum[tid + stride];
      shm_sum_sq[tid] += shm_sum_sq[tid + stride];
    }
    __syncthreads();
  }
  accscalar_t block_sum = shm_sum[0];
  accscalar_t block_sum_sq = shm_sum_sq[0];

  // Atomically add the block's partial sums to the global accumulators for this instance
  if (tid == 0) {
    atomicAdd(&g_sum[instance_idx], block_sum);
    atomicAdd(&g_sum_sq[instance_idx], block_sum_sq);
  }

  // Ensure all blocks have completed their atomic additions
  grid.sync();

  // Phase 2: One designated thread per instance computes the final mean and inv_std
  if (block_in_inst == 0 && tid == 0) {
    accscalar_t total_sum = g_sum[instance_idx];
    accscalar_t total_sum_sq = g_sum_sq[instance_idx];
    accscalar_t mean = total_sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = total_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    accscalar_t inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
    g_mean[instance_idx] = mean;
    g_inv_std[instance_idx] = inv_std;
  }

  // Wait until mean and inv_std are computed
  grid.sync();

  // Phase 3: Each block normalizes its assigned segment using the computed mean and inv_std
  accscalar_t mean = g_mean[instance_idx];
  accscalar_t inv_std = g_inv_std[instance_idx];
  for (int i = seg_start + tid; i < seg_end; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    accscalar_t norm_val = (val - mean) * inv_std;
    // Use __ldg for coalesced read of weight and bias
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}


// C++ interface function. This function sets up the cooperative kernel launch and allocates temporary
// global buffers to accumulate partial sums across blocks for each instance.

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // To evenly distribute work, we choose a number of blocks per instance (tunable parameter).
  int blocks_per_inst = 4;  // For example, 4 blocks per instance
  int total_blocks = outer_size * blocks_per_inst;
  int threads = 256;
  
  // Allocate temporary global buffers for sums, squared sums, and final mean/inv_std values
  auto g_sum = torch::zeros({outer_size}, x.options());
  auto g_sum_sq = torch::zeros({outer_size}, x.options());
  auto g_mean = torch::zeros({outer_size}, x.options());
  auto g_inv_std = torch::zeros({outer_size}, x.options());

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_bytes = threads * 2 * sizeof(accscalar_t);
    // Launch the kernel cooperatively. Note: This kernel requires a cooperative launch configuration.
    layernorm_distributed_kernel<scalar_t><<<total_blocks, threads, shared_bytes>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        output.data_ptr<scalar_t>(),
        normalized_size,
        outer_size,
        blocks_per_inst,
        reinterpret_cast<accscalar_t*>(g_sum.data_ptr<scalar_t>()),
        reinterpret_cast<accscalar_t*>(g_sum_sq.data_ptr<scalar_t>()),
        reinterpret_cast<accscalar_t*>(g_mean.data_ptr<scalar_t>()),
        reinterpret_cast<accscalar_t*>(g_inv_std.data_ptr<scalar_t>()));
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with distributed workload using cooperative groups",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
