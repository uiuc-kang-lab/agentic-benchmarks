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

template <typename scalar_t, int VEC_SIZE>
__global__ void log_softmax_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim_size) {

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int warps_per_block = blockDim.x / 32;
    
    const scalar_t* __restrict__ input_row = input + batch_idx * dim_size;
    scalar_t* output_row = output + batch_idx * dim_size;

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* warp_max = reinterpret_cast<scalar_t*>(smem);
    scalar_t* warp_sum = warp_max + warps_per_block;

    // Vectorized max reduction
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int vec_dim = dim_size / VEC_SIZE;
    for (int idx = tid; idx < vec_dim; idx += blockDim.x) {
        scalar_t vals[VEC_SIZE];
        *reinterpret_cast<typename std::conditional<VEC_SIZE == 4, float4, double2>::type*>(vals) = 
            *reinterpret_cast<const typename std::conditional<VEC_SIZE == 4, float4, double2>::type*>(input_row + idx * VEC_SIZE);
        
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            max_val = max(max_val, vals[i]);
        }
    }
    // Handle remaining elements
    for (int idx = vec_dim * VEC_SIZE + tid; idx < dim_size; idx += blockDim.x) {
        max_val = max(max_val, __ldg(input_row + idx));
    }

    max_val = warp_reduce_max(max_val);
    if (lane_id == 0) warp_max[warp_id] = max_val;
    __syncthreads();

    if (tid < 32) {
        max_val = (tid < warps_per_block) ? warp_max[tid] : -std::numeric_limits<scalar_t>::infinity();
        max_val = warp_reduce_max(max_val);
        if (tid == 0) warp_max[0] = max_val;
    }
    __syncthreads();
    max_val = warp_max[0];

    // Vectorized sum reduction
    scalar_t sum = 0;
    for (int idx = tid; idx < vec_dim; idx += blockDim.x) {
        scalar_t vals[VEC_SIZE];
        *reinterpret_cast<typename std::conditional<VEC_SIZE == 4, float4, double2>::type*>(vals) = 
            *reinterpret_cast<const typename std::conditional<VEC_SIZE == 4, float4, double2>::type*>(input_row + idx * VEC_SIZE);
        
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            sum += exp(vals[i] - max_val);
        }
    }
    // Handle remaining elements
    for (int idx = vec_dim * VEC_SIZE + tid; idx < dim_size; idx += blockDim.x) {
        sum += exp(__ldg(input_row + idx) - max_val);
    }

    sum = warp_reduce_sum(sum);
    if (lane_id == 0) warp_sum[warp_id] = sum;
    __syncthreads();

    if (tid < 32) {
        sum = (tid < warps_per_block) ? warp_sum[tid] : 0;
        sum = warp_reduce_sum(sum);
        if (tid == 0) warp_sum[0] = sum;
    }
    __syncthreads();
    scalar_t log_sum = log(warp_sum[0]);

    // Vectorized write output
    for (int idx = tid; idx < vec_dim; idx += blockDim.x) {
        scalar_t out_vals[VEC_SIZE];
        const scalar_t* in_ptr = input_row + idx * VEC_SIZE;
        
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            out_vals[i] = (__ldg(in_ptr + i) - max_val) - log_sum;
        }
        
        *reinterpret_cast<typename std::conditional<VEC_SIZE == 4, float4, double2>::type*>(output_row + idx * VEC_SIZE) = 
            *reinterpret_cast<typename std::conditional<VEC_SIZE == 4, float4, double2>::type*>(out_vals);
    }
    // Handle remaining elements
    for (int idx = vec_dim * VEC_SIZE + tid; idx < dim_size; idx += blockDim.x) {
        output_row[idx] = (__ldg(input_row + idx) - max_val) - log_sum;
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

    const int threads = std::min(1024, static_cast<int>(((dim_size + 31)/32)*32));
    const bool use_float4 = input.scalar_type() == torch::kFloat32 && dim_size % 4 == 0;
    const bool use_double2 = input.scalar_type() == torch::kFloat64 && dim_size % 2 == 0;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
        size_t shared_mem_size = (threads/32) * sizeof(scalar_t) * 2;
        if (use_float4) {
            log_softmax_forward_kernel<scalar_t, 4><<<batch_size, threads, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else if (use_double2) {
            log_softmax_forward_kernel<scalar_t, 2><<<batch_size, threads, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        } else {
            log_softmax_forward_kernel<scalar_t, 1><<<batch_size, threads, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dim_size);
        }
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
