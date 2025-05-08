/*
Efficient L2 Normalization CUDA Kernel
This kernel combines vectorized loads/stores for contiguous memory (stride_C == 1) and modular reduction using warp-level primitives.
It supports both float and double types and falls back to standard iteration for non-contiguous memory.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction
template <typename scalar_t>
__device__ inline scalar_t warp_reduce_sum(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction
template <typename scalar_t>
__device__ inline scalar_t block_reduce_sum(scalar_t val) {
    __shared__ scalar_t shared[32];  // one per warp, assuming blockDim.x <= 1024
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;  // threadIdx.x / 32
    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    // Only first warp loads the partial sums
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : static_cast<scalar_t>(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// Combined L2 normalization kernel
template <typename scalar_t>
__global__ void l2_normalize_kernel_combined(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int base = vector_idx * outer_stride;
    scalar_t local_sum = 0;

    // Sum-of-squares calculation
    if (stride_C == 1) {
        // Use vectorized memory accesses if contiguous
        if constexpr (sizeof(scalar_t) == 4) { // float
            int aligned = (C / 4) * 4;
            int num_vec = aligned / 4;
            // Vectorized loads using float4
            for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
                const float4* in_vec = reinterpret_cast<const float4*>(input + base);
                float4 vec = in_vec[i];
                local_sum += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
            }
            // Process remaining elements
            for (int i = aligned + threadIdx.x; i < C; i += blockDim.x) {
                scalar_t val = input[base + i];
                local_sum += val * val;
            }
        } else { // double
            int aligned = (C / 2) * 2;
            int num_vec = aligned / 2;  // Process two doubles at a time with double2
            for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
                const double2* in_vec = reinterpret_cast<const double2*>(input + base);
                double2 vec = in_vec[i];
                local_sum += vec.x * vec.x + vec.y * vec.y;
            }
            // Process any remaining elements
            for (int i = aligned + threadIdx.x; i < C; i += blockDim.x) {
                scalar_t val = input[base + i];
                local_sum += val * val;
            }
        }
    } else {
        // Fallback for non-contiguous memory: use every stride_C-th element
        for (int i = threadIdx.x; i < C; i += blockDim.x) {
            scalar_t val = input[base + i * stride_C];
            local_sum += val * val;
        }
    }

    // Block-wide reduction
    scalar_t block_sum = block_reduce_sum(local_sum);

    // Compute inverse L2 norm
    scalar_t inv_norm = 1.0 / (sqrt(block_sum) + static_cast<scalar_t>(1e-12));

    // Write back normalized values
    if (stride_C == 1) {
        if constexpr (sizeof(scalar_t) == 4) { // float
            int aligned = (C / 4) * 4;
            int num_vec = aligned / 4;
            // Vectorized stores using float4
            for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
                float4* out_vec = reinterpret_cast<float4*>(output + base);
                const float4* in_vec = reinterpret_cast<const float4*>(input + base);
                float4 vec = in_vec[i];
                vec.x *= inv_norm;
                vec.y *= inv_norm;
                vec.z *= inv_norm;
                vec.w *= inv_norm;
                out_vec[i] = vec;
            }
            // Tail elements
            for (int i = aligned + threadIdx.x; i < C; i += blockDim.x) {
                scalar_t val = input[base + i];
                output[base + i] = val * inv_norm;
            }
        } else { // double
            int aligned = (C / 2) * 2;
            int num_vec = aligned / 2;
            for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
                double2* out_vec = reinterpret_cast<double2*>(output + base);
                const double2* in_vec = reinterpret_cast<const double2*>(input + base);
                double2 vec = in_vec[i];
                vec.x *= inv_norm;
                vec.y *= inv_norm;
                out_vec[i] = vec;
            }
            for (int i = aligned + threadIdx.x; i < C; i += blockDim.x) {
                scalar_t val = input[base + i];
                output[base + i] = val * inv_norm;
            }
        }
    } else {
        // Fallback for non-contiguous stores
        for (int i = threadIdx.x; i < C; i += blockDim.x) {
            scalar_t val = input[base + i * stride_C];
            output[base + i * stride_C] = val * inv_norm;
        }
    }
}

// Host wrapper
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize_combined", ([&] {
        l2_normalize_kernel_combined<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C, total_vectors, stride_C, outer_stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient L2 normalization with vectorized loads/stores and warp-level reduction");
}
