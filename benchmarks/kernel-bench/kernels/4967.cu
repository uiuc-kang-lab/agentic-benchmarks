#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void l2norm_warp_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int tid = threadIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int warps_per_block = blockDim.x / warpSize;
    const int base = vector_idx * outer_stride;

    // Each thread accumulates its portion
    scalar_t thread_sum = 0;

    if (stride_C == 1) {
        // Use vectorized loads for coalesced memory access
        using Vec4 = typename std::conditional<sizeof(scalar_t) == 4, float4, double2>::type;
        const int vec_size = sizeof(Vec4) / sizeof(scalar_t);
        const Vec4* vec_input = reinterpret_cast<const Vec4*>(input + base);
        const int vec_elements = C / vec_size;
        
        // Process vector elements
        for (int i = tid; i < vec_elements; i += blockDim.x) {
            Vec4 v = vec_input[i];
            if constexpr (sizeof(scalar_t) == 4) {
                thread_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            } else {
                thread_sum += v.x * v.x + v.y * v.y;
            }
        }

        // Handle remaining elements
        for (int i = vec_elements * vec_size + tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i];
            thread_sum += val * val;
        }
    } else {
        // Non-contiguous case
        for (int i = tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i * stride_C];
            thread_sum += val * val;
        }
    }

    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);

    // First thread in each warp stores the result
    __shared__ scalar_t warp_results[32];  // Max warps per block
    if (lane_id == 0) {
        warp_results[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces the warp results
    if (warp_id == 0) {
        thread_sum = (lane_id < warps_per_block) ? warp_results[lane_id] : 0;
        thread_sum = warp_reduce_sum(thread_sum);

        if (lane_id == 0) {
            warp_results[0] = thread_sum;
        }
    }
    __syncthreads();

    // Compute inverse norm once
    const scalar_t inv_norm = 1.0 / (sqrt(warp_results[0]) + 1e-12);

    // Normalize using the same vectorized approach
    if (stride_C == 1) {
        using Vec4 = typename std::conditional<sizeof(scalar_t) == 4, float4, double2>::type;
        const int vec_size = sizeof(Vec4) / sizeof(scalar_t);
        Vec4* vec_output = reinterpret_cast<Vec4*>(output + base);
        const Vec4* vec_input = reinterpret_cast<const Vec4*>(input + base);
        const int vec_elements = C / vec_size;

        // Process vector elements
        for (int i = tid; i < vec_elements; i += blockDim.x) {
            Vec4 v = vec_input[i];
            if constexpr (sizeof(scalar_t) == 4) {
                v.x *= inv_norm;
                v.y *= inv_norm;
                v.z *= inv_norm;
                v.w *= inv_norm;
            } else {
                v.x *= inv_norm;
                v.y *= inv_norm;
            }
            vec_output[i] = v;
        }

        // Handle remaining elements
        for (int i = vec_elements * vec_size + tid; i < C; i += blockDim.x) {
            output[base + i] = input[base + i] * inv_norm;
        }
    } else {
        // Non-contiguous case
        for (int i = tid; i < C; i += blockDim.x) {
            output[base + i * stride_C] = input[base + i * stride_C] * inv_norm;
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

    auto output = torch::empty_like(input);

    // Use multiple of warp size for thread count
    const int threads = 256;  // 8 warps per block
    const int blocks = total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_warp", ([&] {
        l2norm_warp_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &forward, "L2 normalization optimized with warp-level operations");
}