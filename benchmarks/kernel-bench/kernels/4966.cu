#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel that uses __ldg() for read-only memory accesses and ensures 128-bit aligned loads/stores
// It computes the L2 norm for each vector (assumed along dim=1) and normalizes it.

template <typename scalar_t>
__global__ void l2norm_ldg_aligned_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;
    
    // Base offset for this vector
    int base = vector_idx * outer_stride;
    int tid = threadIdx.x;
    scalar_t sum = 0;

    // If the input is contiguous along dim=1, we can use 128-bit aligned vectorized loads.
    if (stride_C == 1) {
        // 128-bit load: for float use float4 (4x32 bits), for double use double2 (2x64 bits)
        const int factor = (sizeof(scalar_t) == 4) ? 4 : 2;
        int aligned_end = (C / factor) * factor;
        
        if constexpr (sizeof(scalar_t) == 4) {
            const float4* in_vec = reinterpret_cast<const float4*>(input + base);
            int num_vec = aligned_end / 4;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                // Using __ldg() for efficient read-only access
                float4 v = __ldg(&in_vec[i]);
                sum += static_cast<scalar_t>(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
            }
        } else {
            const double2* in_vec = reinterpret_cast<const double2*>(input + base);
            int num_vec = aligned_end / 2;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                double2 v = __ldg(&in_vec[i]);
                sum += static_cast<scalar_t>(v.x * v.x + v.y * v.y);
            }
        }
        // Process any tail elements that don't fit into a 128-bit chunk
        for (int i = aligned_end + tid; i < C; i += blockDim.x) {
            scalar_t val = __ldg(&input[base + i]);
            sum += val * val;
        }
    } else {
        // Fallback for non-contiguous cases
        for (int i = tid; i < C; i += blockDim.x) {
            scalar_t val = __ldg(&input[base + i * stride_C]);
            sum += val * val;
        }
    }

    // Intra-warp reduction using warp shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Use shared memory to reduce across warps
    __shared__ scalar_t shmem[32];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shmem[warpId] = sum;
    }
    __syncthreads();
    if (warpId == 0) {
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shmem[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    // Broadcast the inverse norm to all threads
    __shared__ scalar_t inv_norm_shared;
    if (lane == 0) {
        scalar_t norm = sqrt(sum) + static_cast<scalar_t>(1e-12);
        inv_norm_shared = static_cast<scalar_t>(1.0) / norm;
    }
    __syncthreads();
    scalar_t inv_norm = inv_norm_shared;

    // Write back the normalized vector using vectorized stores if possible
    if (stride_C == 1) {
        const int factor = (sizeof(scalar_t) == 4) ? 4 : 2;
        int aligned_end = (C / factor) * factor;
        if constexpr (sizeof(scalar_t) == 4) {
            const float4* in_vec = reinterpret_cast<const float4*>(input + base);
            float4* out_vec = reinterpret_cast<float4*>(output + base);
            int num_vec = aligned_end / 4;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                float4 v = __ldg(&in_vec[i]);
                v.x *= inv_norm;
                v.y *= inv_norm;
                v.z *= inv_norm;
                v.w *= inv_norm;
                out_vec[i] = v;
            }
        } else {
            const double2* in_vec = reinterpret_cast<const double2*>(input + base);
            double2* out_vec = reinterpret_cast<double2*>(output + base);
            int num_vec = aligned_end / 2;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                double2 v = __ldg(&in_vec[i]);
                v.x *= inv_norm;
                v.y *= inv_norm;
                out_vec[i] = v;
            }
        }
        // Process remaining elements
        for (int i = aligned_end + tid; i < C; i += blockDim.x) {
            scalar_t val = __ldg(&input[base + i]);
            output[base + i] = val * inv_norm;
        }
    } else {
        // Fallback for non-contiguous output
        for (int i = tid; i < C; i += blockDim.x) {
            scalar_t val = __ldg(&input[base + i * stride_C]);
            output[base + i * stride_C] = val * inv_norm;
        }
    }
}

// Host function to launch the kernel

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);
    const int threads = 256;
    dim3 blocks(total_vectors);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_ldg_aligned", ([&] {
        l2norm_ldg_aligned_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization using __ldg for read-only accesses and 128-bit aligned operations");
}
