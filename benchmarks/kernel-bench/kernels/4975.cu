#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Shared memory reduction for both small and large vectors
template<typename scalar_t>
__device__ scalar_t block_reduce_sum(scalar_t val, int tid) {
    __shared__ scalar_t shared[32];
    int lane = tid % 32;
    int wid = tid / 32;

    // Warp reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Final reduction across warps
    if (wid == 0) {
        val = (lane < (blockDim.x + 31)/32) ? shared[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Vectorized load function
template<typename scalar_t>
__device__ scalar_t vectorized_load_square_sum(const scalar_t* input, int idx) {
    scalar_t sum = 0;
    if constexpr (sizeof(scalar_t) == 4) {
        float4 v = reinterpret_cast<const float4*>(input)[idx];
        sum = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    } else {
        double2 v = reinterpret_cast<const double2*>(input)[idx];
        sum = v.x * v.x + v.y * v.y;
    }
    return sum;
}

// Vectorized store function
template<typename scalar_t>
__device__ void vectorized_store_normalized(scalar_t* output, const scalar_t* input, 
                                          int idx, scalar_t inv_norm) {
    if constexpr (sizeof(scalar_t) == 4) {
        float4 v = reinterpret_cast<const float4*>(input)[idx];
        v.x *= inv_norm; v.y *= inv_norm;
        v.z *= inv_norm; v.w *= inv_norm;
        reinterpret_cast<float4*>(output)[idx] = v;
    } else {
        double2 v = reinterpret_cast<const double2*>(input)[idx];
        v.x *= inv_norm; v.y *= inv_norm;
        reinterpret_cast<double2*>(output)[idx] = v;
    }
}

template <typename scalar_t>
__global__ void adaptive_l2_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ partial_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const bool is_large_vector) {

    const int tid = threadIdx.x;
    const int vector_idx = blockIdx.x;
    const int base = vector_idx * outer_stride;
    
    scalar_t sum = 0;

    if (stride_C == 1) {
        // Vectorized path
        const int vec_size = (sizeof(scalar_t) == 4) ? 4 : 2;
        const int aligned_end = (C / vec_size) * vec_size;
        const int num_vec = aligned_end / vec_size;

        // Vectorized loads
        for (int i = tid; i < num_vec; i += blockDim.x) {
            sum += vectorized_load_square_sum(input + base, i);
        }

        // Handle remaining elements
        for (int i = aligned_end + tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i];
            sum += val * val;
        }
    } else {
        // Non-contiguous path
        for (int i = tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i * stride_C];
            sum += val * val;
        }
    }

    // Block reduction
    sum = block_reduce_sum(sum, tid);

    if (tid == 0) {
        if (is_large_vector) {
            atomicAdd(&partial_sums[vector_idx], sum);
        } else {
            scalar_t norm = sqrt(sum) + scalar_t(1e-12);
            partial_sums[vector_idx] = norm;
        }
    }
    __syncthreads();

    // For small vectors, normalize immediately
    if (!is_large_vector) {
        scalar_t inv_norm = scalar_t(1.0) / partial_sums[vector_idx];

        if (stride_C == 1) {
            const int vec_size = (sizeof(scalar_t) == 4) ? 4 : 2;
            const int aligned_end = (C / vec_size) * vec_size;
            const int num_vec = aligned_end / vec_size;

            for (int i = tid; i < num_vec; i += blockDim.x) {
                vectorized_store_normalized(output + base, input + base, i, inv_norm);
            }

            for (int i = aligned_end + tid; i < C; i += blockDim.x) {
                output[base + i] = input[base + i] * inv_norm;
            }
        } else {
            for (int i = tid; i < C; i += blockDim.x) {
                output[base + i * stride_C] = input[base + i * stride_C] * inv_norm;
            }
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);
    
    const int THRESHOLD = 1024;
    const bool is_large_vector = C > THRESHOLD;
    const int threads = 256;
    const int blocks = total_vectors;

    auto output = torch::empty_like(input);
    auto partial_sums = torch::empty({total_vectors}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_l2_norm", ([&] {
        adaptive_l2_norm_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            C, total_vectors, stride_C, outer_stride,
            is_large_vector
        );

        if (is_large_vector) {
            adaptive_l2_norm_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                partial_sums.data_ptr<scalar_t>(),
                C, total_vectors, stride_C, outer_stride,
                false  // Second pass for normalization
            );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive L2 normalization");
}