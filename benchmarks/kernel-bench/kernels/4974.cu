#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void l2norm_optimized_kernel(
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
    
    using Vector4 = typename std::conditional<sizeof(scalar_t) == 4, float4, double2>::type;
    const int vector_size = sizeof(Vector4) / sizeof(scalar_t);
    
    scalar_t sum = 0;
    
    if (stride_C == 1) {
        const Vector4* vec_input = reinterpret_cast<const Vector4*>(input + base);
        const int vec_elements = C / vector_size;
        
        #pragma unroll 4
        for (int i = tid; i < vec_elements; i += blockDim.x) {
            Vector4 vec = vec_input[i];
            if constexpr (sizeof(scalar_t) == 4) {
                sum += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
            } else {
                sum += vec.x * vec.x + vec.y * vec.y;
            }
        }
        
        for (int i = C - (C % vector_size) + tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i];
            sum += val * val;
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i * stride_C];
            sum += val * val;
        }
    }

    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ scalar_t warp_sums[32];
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane_id] : 0;
        
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            warp_sums[0] = rsqrt(sum + scalar_t(1e-12));
        }
    }

    __syncthreads();
    
    const scalar_t inv_norm = warp_sums[0];

    if (stride_C == 1) {
        Vector4* vec_output = reinterpret_cast<Vector4*>(output + base);
        const Vector4* vec_input = reinterpret_cast<const Vector4*>(input + base);
        const int vec_elements = C / vector_size;

        #pragma unroll 4
        for (int i = tid; i < vec_elements; i += blockDim.x) {
            Vector4 vec = vec_input[i];
            if constexpr (sizeof(scalar_t) == 4) {
                vec.x *= inv_norm;
                vec.y *= inv_norm;
                vec.z *= inv_norm;
                vec.w *= inv_norm;
            } else {
                vec.x *= inv_norm;
                vec.y *= inv_norm;
            }
            vec_output[i] = vec;
        }

        for (int i = C - (C % vector_size) + tid; i < C; i += blockDim.x) {
            output[base + i] = input[base + i] * inv_norm;
        }
    } else {
        #pragma unroll 4
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

    const int threads = 256;
    const int blocks = total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_optimized", ([&] {
        l2norm_optimized_kernel<scalar_t><<<blocks, threads>>>(
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