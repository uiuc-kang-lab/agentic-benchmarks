#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void l2_normalize_adaptive_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ global_sum,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const bool use_single_block) {

    const int vector_idx = blockIdx.x / (use_single_block ? 1 : 2);
    if (vector_idx >= total_vectors) return;

    const int phase = use_single_block ? 0 : (blockIdx.x % 2);
    const int base = vector_idx * outer_stride;
    const int tid = threadIdx.x;

    if (phase == 0) {  // Compute sum phase
        scalar_t sum = 0;

        if (stride_C == 1) {
            // Vectorized loads for contiguous data
            const int aligned_end = (C / 4) * 4;
            const int num_vec = aligned_end / 4;
            
            if constexpr (sizeof(scalar_t) == 4) {
                for (int i = tid; i < num_vec; i += blockDim.x) {
                    const float4* in_vec = reinterpret_cast<const float4*>(input + base);
                    float4 vec = in_vec[i];
                    sum += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
                }
            } else {
                for (int i = tid; i < C/2; i += blockDim.x) {
                    const double2* in_vec = reinterpret_cast<const double2*>(input + base);
                    double2 vec = in_vec[i];
                    sum += vec.x * vec.x + vec.y * vec.y;
                }
            }

            // Handle remaining elements
            for (int i = aligned_end + tid; i < C; i += blockDim.x) {
                scalar_t val = input[base + i];
                sum += val * val;
            }
        } else {
            for (int i = tid; i < C; i += blockDim.x) {
                scalar_t val = input[base + i * stride_C];
                sum += val * val;
            }
        }

        // Warp reduction
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (tid % 32 == 0)
            atomicAdd(&global_sum[vector_idx], sum);

        if (use_single_block) {
            __syncthreads();
            
            // Normalize phase (same block)
            scalar_t norm = sqrt(global_sum[vector_idx]) + 1e-12;
            scalar_t inv_norm = 1.0 / norm;

            if (stride_C == 1) {
                const int aligned_end = (C / 4) * 4;
                if constexpr (sizeof(scalar_t) == 4) {
                    for (int i = tid; i < aligned_end/4; i += blockDim.x) {
                        float4* out_vec = reinterpret_cast<float4*>(output + base);
                        const float4* in_vec = reinterpret_cast<const float4*>(input + base);
                        float4 vec = in_vec[i];
                        vec.x *= inv_norm;
                        vec.y *= inv_norm;
                        vec.z *= inv_norm;
                        vec.w *= inv_norm;
                        out_vec[i] = vec;
                    }
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
    } else {  // Normalize phase (separate block)
        scalar_t norm = sqrt(global_sum[vector_idx]) + 1e-12;
        scalar_t inv_norm = 1.0 / norm;

        if (stride_C == 1) {
            const int aligned_end = (C / 4) * 4;
            if constexpr (sizeof(scalar_t) == 4) {
                for (int i = tid; i < aligned_end/4; i += blockDim.x) {
                    float4* out_vec = reinterpret_cast<float4*>(output + base);
                    const float4* in_vec = reinterpret_cast<const float4*>(input + base);
                    float4 vec = in_vec[i];
                    vec.x *= inv_norm;
                    vec.y *= inv_norm;
                    vec.z *= inv_norm;
                    vec.w *= inv_norm;
                    out_vec[i] = vec;
                }
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

    auto output = torch::empty_like(input);
    auto global_sum = torch::zeros({total_vectors}, input.options());

    const int threads = 256;
    const bool use_single_block = (C <= 4096);  // Threshold for single-block approach
    const int blocks = total_vectors * (use_single_block ? 1 : 2);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_adaptive", ([&] {
        l2_normalize_adaptive_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            global_sum.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride,
            use_single_block
        );
    }));

    return output;
}