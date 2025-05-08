#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t, bool IsContiguous>
__device__ scalar_t calculate_sum_of_squares(
    const scalar_t* input,
    const int base_offset,
    const int C,
    const int stride_C) {
    scalar_t sum = 0.0;
    
    if constexpr (IsContiguous) {
        const int aligned_end = (C / 4) * 4;
        const int num_vec = aligned_end / 4;
        
        if constexpr (sizeof(scalar_t) == 4) {
            const float4* in_vec = reinterpret_cast<const float4*>(input + base_offset);
            for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
                float4 vec = in_vec[i];
                sum += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
            }
            for (int i = aligned_end + threadIdx.x; i < C; i += blockDim.x) {
                scalar_t val = input[base_offset + i];
                sum += val * val;
            }
        } else {
            const double2* in_vec = reinterpret_cast<const double2*>(input + base_offset);
            const int num_vec2 = C / 2;
            for (int i = threadIdx.x; i < num_vec2/2; i += blockDim.x) {
                double2 vec = in_vec[i];
                sum += vec.x * vec.x + vec.y * vec.y;
            }
            for (int i = (num_vec2 * 2) + threadIdx.x; i < C; i += blockDim.x) {
                scalar_t val = input[base_offset + i];
                sum += val * val;
            }
        }
    } else {
        for (int k = threadIdx.x; k < C; k += blockDim.x) {
            const scalar_t val = input[base_offset + k * stride_C];
            sum += val * val;
        }
    }
    return sum;
}

template <typename scalar_t>
__device__ scalar_t block_reduce_sum(scalar_t local_sum) {
    __shared__ scalar_t shared[256];
    const int lane = threadIdx.x % 32;
    const int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) shared[wid] = local_sum;
    __syncthreads();

    scalar_t block_sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) shared[0] = block_sum;
    }
    __syncthreads();
    
    return shared[0];
}

template <typename scalar_t, bool IsContiguous>
__device__ void normalize_vector(
    const scalar_t* input,
    scalar_t* output,
    const int base_offset,
    const int C,
    const int stride_C,
    const scalar_t inv_norm) {
    
    if constexpr (IsContiguous) {
        const int aligned_end = (C / 4) * 4;
        const int num_vec = aligned_end / 4;

        if constexpr (sizeof(scalar_t) == 4) {
            float4* out_vec = reinterpret_cast<float4*>(output + base_offset);
            const float4* in_vec = reinterpret_cast<const float4*>(input + base_offset);
            
            for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
                float4 vec = in_vec[i];
                vec.x *= inv_norm;
                vec.y *= inv_norm;
                vec.z *= inv_norm;
                vec.w *= inv_norm;
                out_vec[i] = vec;
            }
            
            for (int i = aligned_end + threadIdx.x; i < C; i += blockDim.x) {
                output[base_offset + i] = input[base_offset + i] * inv_norm;
            }
        } else {
            double2* out_vec = reinterpret_cast<double2*>(output + base_offset);
            const double2* in_vec = reinterpret_cast<const double2*>(input + base_offset);
            const int num_vec2 = C / 2;
            
            for (int i = threadIdx.x; i < num_vec2/2; i += blockDim.x) {
                double2 vec = in_vec[i];
                vec.x *= inv_norm;
                vec.y *= inv_norm;
                out_vec[i] = vec;
            }
            
            for (int i = (num_vec2 * 2) + threadIdx.x; i < C; i += blockDim.x) {
                output[base_offset + i] = input[base_offset + i] * inv_norm;
            }
        }
    } else {
        for (int k = threadIdx.x; k < C; k += blockDim.x) {
            output[base_offset + k * stride_C] = input[base_offset + k * stride_C] * inv_norm;
        }
    }
}

template <typename scalar_t>
__global__ void l2_normalize_kernel_optimized(
    const scalar_t* input,
    scalar_t* output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;
    
    scalar_t sum = (stride_C == 1) ?
        calculate_sum_of_squares<scalar_t, true>(input, base_offset, C, stride_C) :
        calculate_sum_of_squares<scalar_t, false>(input, base_offset, C, stride_C);
    
    sum = block_reduce_sum(sum);
    
    const scalar_t inv_norm = 1.0 / (sqrt(sum) + 1e-12);
    
    if (stride_C == 1) {
        normalize_vector<scalar_t, true>(input, output, base_offset, C, stride_C, inv_norm);
    } else {
        normalize_vector<scalar_t, false>(input, output, base_offset, C, stride_C, inv_norm);
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

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize_optimized", [&] {
        l2_normalize_kernel_optimized<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C, total_vectors, stride_C, outer_stride
        );
    });

    return output;
}