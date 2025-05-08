#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warps_per_block = blockDim.x / 32;
    
    // Align input/output pointers to warp size
    const scalar_t* __restrict__ input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* warp_max = reinterpret_cast<scalar_t*>(smem);
    scalar_t* warp_sum = warp_max + warps_per_block;

    // Process elements in a coalesced manner within each warp
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Each thread processes elements with a stride of warp size
    #pragma unroll 4
    for (int base_idx = warp_id * 32; base_idx < dim_size; base_idx += blockDim.x) {
        int idx = base_idx + lane_id;
        if (idx < dim_size) {
            max_val = max(max_val, __ldg(input_row + idx));
        }
    }

    max_val = warp_reduce_max(max_val);
    if (lane_id == 0) {
        warp_max[warp_id] = max_val;
    }
    __syncthreads();

    if (tid < 32) {
        max_val = (tid < warps_per_block) ? warp_max[tid] : -std::numeric_limits<scalar_t>::infinity();
        max_val = warp_reduce_max(max_val);
        if (tid == 0) {
            warp_max[0] = max_val;
        }
    }
    __syncthreads();
    
    max_val = warp_max[0];

    // Compute exp sum with coalesced access pattern
    scalar_t sum = 0;
    #pragma unroll 4
    for (int base_idx = warp_id * 32; base_idx < dim_size; base_idx += blockDim.x) {
        int idx = base_idx + lane_id;
        if (idx < dim_size) {
            sum += exp(__ldg(input_row + idx) - max_val);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane_id == 0) {
        warp_sum[warp_id] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        sum = (tid < warps_per_block) ? warp_sum[tid] : 0;
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            warp_sum[0] = sum;
        }
    }
    __syncthreads();

    scalar_t log_sum = log(warp_sum[0]);

    // Write output with coalesced access pattern
    #pragma unroll 4
    for (int base_idx = warp_id * 32; base_idx < dim_size; base_idx += blockDim.x) {
        int idx = base_idx + lane_id;
        if (idx < dim_size) {
            output_row[idx] = (__ldg(input_row + idx) - max_val) - log_sum;
        }
    }
}

torch::Tensor log_softmax_cuda_forward(torch::Tensor input, int64_t dim) {
    auto ndim = input.dim();
    dim = dim >= 0 ? dim : dim + ndim;

    std::vector<int64_t> permute_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) permute_dims.push_back(i);
    }
    permute_dims.push_back(dim);

    input = input.permute(permute_dims).contiguous();
    auto output = torch::empty_like(input);
    
    int64_t batch_size = input.numel() / input.size(-1);
    int64_t dim_size = input.size(-1);

    // Ensure thread count is multiple of warp size
    const int threads_per_block = 256;
    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
        size_t shared_mem_size = (threads_per_block/32) * sizeof(scalar_t) * 2;
        log_softmax_forward_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size);
    }));

    std::vector<int64_t> inverse_permute_dims(ndim);
    for (size_t i = 0; i < permute_dims.size(); ++i) {
        inverse_permute_dims[permute_dims[i]] = i;
    }
    return output.permute(inverse_permute_dims);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward (CUDA)");
}