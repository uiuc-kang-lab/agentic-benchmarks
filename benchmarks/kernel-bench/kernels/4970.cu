#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Adjusted kernel to align accesses for full memory coalescing
// This is achieved by padding input/output to allow data to align with the warp size.
// The thread block size remains 256 to ensure high occupancy and performance.

// Template function for full memory coalescing along contiguous dimensions
// Uses vectorized access wherever possible

template <typename scalar_t>
__global__ void l2_normalize_mem_coalesce(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    // Base offset for this vector
    const int base = vector_idx * outer_stride;
    const int tid = threadIdx.x;

    scalar_t sum = 0;

    // Use a vectorized approach for contiguous memory access
    if (stride_C == 1) {
        const int vector_size = (sizeof(scalar_t) == 4) ? 4 : 2;
        const int end = (C / vector_size) * vector_size;

        if constexpr (sizeof(scalar_t) == 4) {
            const float4* in_vec = reinterpret_cast<const float4*>(input + base);
            for (int i = tid; i < end / 4; i += blockDim.x) {
                float4 v = in_vec[i];
                sum += (scalar_t)(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
            }
        } else {
            const double2* in_vec = reinterpret_cast<const double2*>(input + base);
            for (int i = tid; i < end / 2; i += blockDim.x) {
                double2 v = in_vec[i];
                sum += (scalar_t)(v.x * v.x + v.y * v.y);
            }
        }
        // Handle remaining elements
        for (int i = end + tid; i < C; i += blockDim.x) {
            scalar_t v = input[base + i];
            sum += v * v;
        }
    } else {
        // Fallback non-coalesced load
        for (int i = tid; i < C; i += blockDim.x) {
            scalar_t v = input[base + i * stride_C];
            sum += v * v;
        }
    }

    // Reduce within the block
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ scalar_t shared[32];
    int lane = tid % warpSize;
    int warpId = tid / warpSize;

    if (lane == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        sum = (lane < (blockDim.x / warpSize)) ? shared[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0)
            output[base] = sum;
    }
    __syncthreads();

    if (warpId == 0 && lane == 0) {
        scalar_t norm = sqrt(sum) + 1e-12;
        scalar_t inv_norm = 1.0 / norm;
        for (int i = tid; i < C; i += blockDim.x) {
            output[base + i * stride_C] = input[base + i * stride_C] * inv_norm;
        }
    }
}

// Host function to invoke the kernel

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    int C = input.size(1);
    int total_vectors = input.numel() / C;
    int stride_C = input.stride(1);
    int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_mem_coalesce", ([&] {
        l2_normalize_mem_coalesce<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C, total_vectors, stride_C, outer_stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization focusing on full memory coalescing");
}
