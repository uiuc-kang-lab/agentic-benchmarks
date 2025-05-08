#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Unified kernel for L2 normalization, adapting according to the configuration (using vectorized memory operations and multi-phase if necessary)
template <typename scalar_t>
__global__ void efficient_l2_normalize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ global_sum,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int tasks_per_vector,
    const bool single_phase
    ) {

    int vector_idx = blockIdx.x / tasks_per_vector;
    int task_idx = blockIdx.x % tasks_per_vector;
    if (vector_idx >= total_vectors) return;

    int task_length = (C + tasks_per_vector - 1) / tasks_per_vector;  // partition C among tasks
    int start = task_idx * task_length;
    int end = min(start + task_length, C);

    int base = vector_idx * outer_stride;
    int tid = threadIdx.x;
    scalar_t sum = 0;

    if (stride_C == 1) {
        const int factor = (sizeof(scalar_t) == 4) ? 4 : 2;
        int aligned_end = start + ((end - start) / factor) * factor;
        if constexpr (sizeof(scalar_t) == 4) {
            const float4* in_vec = reinterpret_cast<const float4*>(input + base + start);
            int num_vec = (aligned_end - start) / factor;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                float4 v = in_vec[i];
                sum += (scalar_t)(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
            }
        } else {
            const double2* in_vec = reinterpret_cast<const double2*>(input + base + start);
            int num_vec = (aligned_end - start) / factor;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                double2 v = in_vec[i];
                sum += (scalar_t)(v.x * v.x + v.y * v.y);
            }
        }
        // Scalar path for unaligned elements
        for (int i = aligned_end + tid; i < end; i += blockDim.x) {
            scalar_t v = input[base + i];
            sum += v * v;
        }
    } else {
        // Fallback for non-contiguous data
        for (int i = start + tid; i < end; i += blockDim.x) {
            scalar_t v = input[base + i * stride_C];
            sum += v * v;
        }
    }

    // Intra-block reduction
    extern __shared__ scalar_t shared[];
    shared[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (single_phase) {
            // directly compute normalization if single phase
            scalar_t norm = sqrt(shared[0]) + 1e-12;
            scalar_t inv_norm = 1.0 / norm;
            for (int i = start; i < end; i++) {
                output[base + i] = input[base + i] * inv_norm;
            }
        } else {
            // accumulate to global sum otherwise
            atomicAdd(&global_sum[vector_idx], shared[0]);
        }
    }

    if (!single_phase && task_idx == 0) {
        // Phase 2: Normalize if in multi-phase mode
        __syncthreads(); // Ensure global_sum is updated
        scalar_t norm = sqrt(global_sum[vector_idx]) + 1e-12;
        scalar_t inv_norm = 1.0 / norm;
        if (stride_C == 1) {
            const int factor = (sizeof(scalar_t) == 4) ? 4 : 2;
            int aligned_end = (C / factor) * factor;
            if constexpr (sizeof(scalar_t) == 4) {
                float4* out_vec = reinterpret_cast<float4*>(output + base);
                const float4* in_vec = reinterpret_cast<const float4*>(input + base);
                int num_vec = aligned_end / factor;
                for (int i = tid; i < num_vec; i += blockDim.x) {
                    float4 v = in_vec[i];
                    v.x *= inv_norm; v.y *= inv_norm;
                    v.z *= inv_norm; v.w *= inv_norm;
                    out_vec[i] = v;
                }
            } else {
                double2* out_vec = reinterpret_cast<double2*>(output + base);
                const double2* in_vec = reinterpret_cast<const double2*>(input + base);
                int num_vec = aligned_end / factor;
                for (int i = tid; i < num_vec; i += blockDim.x) {
                    double2 v = in_vec[i];
                    v.x *= inv_norm; v.y *= inv_norm;
                    out_vec[i] = v;
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


// Host function
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);
    
    auto output = torch::empty_like(input);
    auto global_sum = torch::zeros({total_vectors}, input.options());

    const int threshold = 1024;
    const int threads = 256;
    bool single_phase = C <= threshold;
    int tasks_per_vector = single_phase ? 1 : max(1, (C + 1023) / 1024);

    int total_blocks = total_vectors * tasks_per_vector;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "efficient_l2_norm", ([&] {
        efficient_l2_normalize_kernel<scalar_t><<<total_blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            global_sum.data_ptr<scalar_t>(),
            C, total_vectors, stride_C, outer_stride, tasks_per_vector, single_phase);
    }));

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient L2 normalization combining strategies for varied dimensional lengths");
}
