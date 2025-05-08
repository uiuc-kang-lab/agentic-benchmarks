#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename scalar_t>
__global__ void l2_normalize_kernel_warp(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int wid = tid / 32;
    const int warps_per_block = blockDim.x / 32;

    // Use float4 for vectorized memory access
    const int vector_size = 4;
    const int C_aligned = C / vector_size * vector_size;
    
    scalar_t sum = 0.0;
    
    // Vector loads for aligned portion
    float4 vec_val;
    for (int k = tid * vector_size; k < C_aligned; k += blockDim.x * vector_size) {
        if (k + vector_size <= C) {
            vec_val = *reinterpret_cast<const float4*>(&input[base_offset + k * stride_C]);
            sum += vec_val.x * vec_val.x + vec_val.y * vec_val.y + 
                   vec_val.z * vec_val.z + vec_val.w * vec_val.w;
        }
    }

    // Handle remaining elements
    for (int k = C_aligned + tid; k < C; k += blockDim.x) {
        const scalar_t val = input[base_offset + k * stride_C];
        sum += val * val;
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Inter-warp reduction using first warp
    if (lane == 0) {
        // Store warp result to registers
        scalar_t warp_sum = sum;
        
        // First thread of first warp reduces all warp sums
        if (wid == 0) {
            for (int w = 1; w < warps_per_block; w++) {
                warp_sum += __shfl_sync(0xffffffff, sum, w * 32);
            }
            sum = warp_sum;
        }
    }

    // Broadcast final sum from first thread to all threads
    scalar_t final_sum = __shfl_sync(0xffffffff, sum, 0);
    
    const scalar_t inv_norm = 1.0 / (sqrt(final_sum) + 1e-12);

    // Vectorized normalization for aligned portion
    for (int k = tid * vector_size; k < C_aligned; k += blockDim.x * vector_size) {
        if (k + vector_size <= C) {
            vec_val = *reinterpret_cast<const float4*>(&input[base_offset + k * stride_C]);
            vec_val.x *= inv_norm;
            vec_val.y *= inv_norm;
            vec_val.z *= inv_norm;
            vec_val.w *= inv_norm;
            *reinterpret_cast<float4*>(&output[base_offset + k * stride_C]) = vec_val;
        }
    }

    // Handle remaining elements
    for (int k = C_aligned + tid; k < C; k += blockDim.x) {
        output[base_offset + k * stride_C] = input[base_offset + k * stride_C] * inv_norm;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize", [&] {
        l2_normalize_kernel_warp<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization along dim=1");
}