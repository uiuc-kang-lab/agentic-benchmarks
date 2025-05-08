#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, bool IsContiguous>
__global__ void l2norm_uniform_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int base = vector_idx * outer_stride;
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    
    // Compute aligned boundaries for vectorized loads
    constexpr int vec_size = sizeof(scalar_t) == 4 ? 4 : 2;
    const int aligned_C = (C / vec_size) * vec_size;
    
    using Vec = typename std::conditional<sizeof(scalar_t) == 4, float4, double2>::type;
    
    scalar_t sum = 0;
    
    if constexpr (IsContiguous) {
        // Vectorized load path for contiguous data
        const Vec* vec_input = reinterpret_cast<const Vec*>(input + base);
        const int vec_elements = aligned_C / vec_size;
        
        // All threads in warp process aligned data together
        #pragma unroll 4
        for (int i = tid; i < vec_elements; i += blockDim.x) {
            Vec v = vec_input[i];
            if constexpr (sizeof(scalar_t) == 4) {
                sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            } else {
                sum += v.x * v.x + v.y * v.y;
            }
        }
        
        // Process remaining elements - all threads in warp either process or skip together
        if (aligned_C < C) {
            #pragma unroll
            for (int i = aligned_C + tid; i < C; i += blockDim.x) {
                scalar_t val = input[base + i];
                sum += val * val;
            }
        }
    } else {
        // Non-contiguous path - ensure uniform access pattern within warp
        #pragma unroll 4
        for (int i = tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i * stride_C];
            sum += val * val;
        }
    }

    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Block-level reduction using shared memory
    __shared__ scalar_t warp_sums[32];  // One element per warp
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane_id] : 0;
        
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    // Broadcast final result to all threads
    __shared__ scalar_t norm_factor;
    if (tid == 0) {
        norm_factor = 1.0 / (sqrt(sum) + 1e-12);
    }
    __syncthreads();

    // Normalize using vectorized operations when possible
    if constexpr (IsContiguous) {
        Vec* vec_output = reinterpret_cast<Vec*>(output + base);
        const Vec* vec_input = reinterpret_cast<const Vec*>(input + base);
        const int vec_elements = aligned_C / vec_size;
        
        #pragma unroll 4
        for (int i = tid; i < vec_elements; i += blockDim.x) {
            Vec v = vec_input[i];
            if constexpr (sizeof(scalar_t) == 4) {
                v.x *= norm_factor;
                v.y *= norm_factor;
                v.z *= norm_factor;
                v.w *= norm_factor;
            } else {
                v.x *= norm_factor;
                v.y *= norm_factor;
            }
            vec_output[i] = v;
        }

        if (aligned_C < C) {
            #pragma unroll
            for (int i = aligned_C + tid; i < C; i += blockDim.x) {
                output[base + i] = input[base + i] * norm_factor;
            }
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < C; i += blockDim.x) {
            output[base + i * stride_C] = input[base + i * stride_C] * norm_factor;
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = total_vectors;

    const bool is_contiguous = (stride_C == 1);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_uniform", ([&] {
        if (is_contiguous) {
            l2norm_uniform_kernel<scalar_t, true><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                C, total_vectors, stride_C, outer_stride);
        } else {
            l2norm_uniform_kernel<scalar_t, false><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                C, total_vectors, stride_C, outer_stride);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with uniform warp execution");
}