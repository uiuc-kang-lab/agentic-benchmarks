/*
Combined L2 normalization kernel

This implementation chooses a strategy based on the size of the normalization dimension (C).
For small C (<= 1024), a single kernel is launched per vector that computes the L2 norm (using vectorized loads/stores and warp-level reduction) and then normalizes the vector.
For large C (> 1024), a two-phase approach is used:
  1) A partial reduction kernel (with multiple blocks per vector) computes segments of the sum-of-squares using vectorized loads when possible, and
     the results are atomically accumulated into a global sum array.
  2) A normalization kernel uses the computed norm to scale each segment of the vector (again employing vectorized stores if applicable).

Fallback paths for non-contiguous accesses are provided.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>


// Single-kernel approach for small C (<= threshold)
template <typename scalar_t>
__global__ void l2_norm_single_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int outer_stride,
    const int stride_C) {

    // Each block processes one vector
    int vector_idx = blockIdx.x;
    if (vector_idx >= gridDim.x) return;
    int base = vector_idx * outer_stride;
    int tid = threadIdx.x;
    scalar_t sum = 0;

    if (stride_C == 1) {
        // Use vectorized loads if possible
        // For float: vector size 4, for double: vector size 2
        const int factor = (sizeof(scalar_t) == 4) ? 4 : 2;
        int aligned_end = (C / factor) * factor;
        if constexpr (sizeof(scalar_t) == 4) {
            const float4* inp = reinterpret_cast<const float4*>(input + base);
            int num_vec = aligned_end / 4;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                float4 v = inp[i];
                sum += (scalar_t)(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
            }
        } else {
            const double2* inp = reinterpret_cast<const double2*>(input + base);
            int num_vec = aligned_end / 2;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                double2 v = inp[i];
                sum += (scalar_t)(v.x * v.x + v.y * v.y);
            }
        }
        // Process remaining elements
        for (int i = aligned_end + tid; i < C; i += blockDim.x) {
            scalar_t v = input[base + i];
            sum += v * v;
        }
    } else {
        // Non-contiguous fallback
        for (int i = tid; i < C; i += blockDim.x) {
            scalar_t v = input[base + i * stride_C];
            sum += v * v;
        }
    }

    // Intra-block reduction using warp shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Use shared memory to accumulate sums from each warp
    __shared__ scalar_t shared[32];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) shared[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    // The first thread computes the normalization and writes back
    if (lane == 0) {
        scalar_t norm = sqrt(sum) + (scalar_t)1e-12;
        scalar_t inv_norm = (scalar_t)1.0 / norm;

        // Normalize vector elements using vectorized stores if possible
        if (stride_C == 1) {
            const int factor = (sizeof(scalar_t) == 4) ? 4 : 2;
            int aligned_end = (C / factor) * factor;
            if constexpr (sizeof(scalar_t) == 4) {
                float4* outp = reinterpret_cast<float4*>(output + base);
                const float4* inp = reinterpret_cast<const float4*>(input + base);
                int num_vec = aligned_end / 4;
                for (int i = 0; i < num_vec; i++) {
                    float4 v = inp[i];
                    v.x *= inv_norm; v.y *= inv_norm;
                    v.z *= inv_norm; v.w *= inv_norm;
                    outp[i] = v;
                }
            } else {
                double2* outp = reinterpret_cast<double2*>(output + base);
                const double2* inp = reinterpret_cast<const double2*>(input + base);
                int num_vec = aligned_end / 2;
                for (int i = 0; i < num_vec; i++) {
                    double2 v = inp[i];
                    v.x *= inv_norm; v.y *= inv_norm;
                    outp[i] = v;
                }
            }
            for (int i = aligned_end; i < C; i++) {
                output[base + i] = input[base + i] * inv_norm;
            }
        } else {
            for (int i = 0; i < C; i++) {
                output[base + i * stride_C] = input[base + i * stride_C] * inv_norm;
            }
        }
    }
}


// Two-phase approach kernels for large C (> threshold)

// Phase 1: Partial reduction kernel with vectorized loads
template <typename scalar_t>
__global__ void l2_norm_partial_kernel_combined(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ global_sum,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int blocks_per_vector) {

    // Each block processes a segment of a vector
    int vector_idx = blockIdx.x / blocks_per_vector;
    int seg_idx = blockIdx.x % blocks_per_vector;
    if (vector_idx >= total_vectors) return;

    // Determine segment boundaries
    int seg_length = (C + blocks_per_vector - 1) / blocks_per_vector;  // ceil division
    int start = seg_idx * seg_length;
    int end = start + seg_length;
    if (end > C) end = C;

    int base = vector_idx * outer_stride;
    int tid = threadIdx.x;
    scalar_t partial = 0;

    if (stride_C == 1) {
        // Vectorized load path
        const int factor = (sizeof(scalar_t) == 4) ? 4 : 2;
        int aligned_end = start + ((end - start) / factor) * factor;
        if constexpr (sizeof(scalar_t) == 4) {
            const float4* in_vec = reinterpret_cast<const float4*>(input + base + start);
            int num_vec = (aligned_end - start) / 4;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                float4 v = in_vec[i];
                partial += (scalar_t)(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
            }
        } else {
            const double2* in_vec = reinterpret_cast<const double2*>(input + base + start);
            int num_vec = (aligned_end - start) / 2;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                double2 v = in_vec[i];
                partial += (scalar_t)(v.x * v.x + v.y * v.y);
            }
        }
        for (int i = aligned_end + tid; i < end; i += blockDim.x) {
            scalar_t v = input[base + i];
            partial += v * v;
        }
    } else {
        // Non-contiguous fallback
        for (int i = start + tid; i < end; i += blockDim.x) {
            scalar_t v = input[base + i * stride_C];
            partial += v * v;
        }
    }

    // Intra-block reduction using shared memory
    __shared__ scalar_t sdata[256];
    sdata[tid] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&global_sum[vector_idx], sdata[0]);
    }
}

// Phase 2: Normalization kernel with vectorized stores
template <typename scalar_t>
__global__ void l2_normalize_kernel_combined(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ global_sum,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int blocks_per_vector) {

    int vector_idx = blockIdx.x / blocks_per_vector;
    int seg_idx = blockIdx.x % blocks_per_vector;
    if (vector_idx >= total_vectors) return;

    int seg_length = (C + blocks_per_vector - 1) / blocks_per_vector;
    int start = seg_idx * seg_length;
    int end = start + seg_length;
    if (end > C) end = C;

    int base = vector_idx * outer_stride;
    scalar_t norm = sqrt(global_sum[vector_idx]) + (scalar_t)1e-12;
    scalar_t inv_norm = (scalar_t)1.0 / norm;

    int tid = threadIdx.x;
    if (stride_C == 1) {
        const int factor = (sizeof(scalar_t) == 4) ? 4 : 2;
        int aligned_end = start + ((end - start) / factor) * factor;
        if constexpr (sizeof(scalar_t) == 4) {
            float4* out_vec = reinterpret_cast<float4*>(output + base + start);
            const float4* in_vec = reinterpret_cast<const float4*>(input + base + start);
            int num_vec = (aligned_end - start) / 4;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                float4 v = in_vec[i];
                v.x *= inv_norm;
                v.y *= inv_norm;
                v.z *= inv_norm;
                v.w *= inv_norm;
                out_vec[i] = v;
            }
        } else {
            double2* out_vec = reinterpret_cast<double2*>(output + base + start);
            const double2* in_vec = reinterpret_cast<const double2*>(input + base + start);
            int num_vec = (aligned_end - start) / 2;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                double2 v = in_vec[i];
                v.x *= inv_norm;
                v.y *= inv_norm;
                out_vec[i] = v;
            }
        }
        for (int i = aligned_end + tid; i < end; i += blockDim.x) {
            output[base + i] = input[base + i] * inv_norm;
        }
    } else {
        for (int i = start + tid; i < end; i += blockDim.x) {
            output[base + i * stride_C] = input[base + i * stride_C] * inv_norm;
        }
    }
}


// Host forward function that selects the optimal strategy based on C
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    // Choose strategy based on vector length
    // Use single-kernel approach if C is small; else use two-phase reduction
    const int threshold = 1024;
    const int threads = 256;

    if (C <= threshold) {
        // Launch one block per vector
        dim3 blocks(total_vectors);
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_single", ([&] {
            l2_norm_single_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                C, outer_stride, stride_C);
        }));
    } else {
        // Two-phase approach for large C
        int seg_size = 1024;  // number of elements each block processes
        int blocks_per_vector = (C + seg_size - 1) / seg_size;
        int total_blocks = total_vectors * blocks_per_vector;

        auto global_sum = torch::zeros({total_vectors}, input.options());

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_partial_combined", ([&] {
            l2_norm_partial_kernel_combined<scalar_t><<<total_blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                global_sum.data_ptr<scalar_t>(),
                C, total_vectors, stride_C, outer_stride, blocks_per_vector);
        }));

        // Launch normalization kernel with same grid configuration
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_normalize_combined", ([&] {
            l2_normalize_kernel_combined<scalar_t><<<total_blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                global_sum.data_ptr<scalar_t>(),
                C, total_vectors, stride_C, outer_stride, blocks_per_vector);
        }));
    }

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization combining vectorized memory accesses and multi-block reduction");
}
