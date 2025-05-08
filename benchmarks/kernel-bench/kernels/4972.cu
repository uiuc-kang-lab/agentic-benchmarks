#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for L2 normalization with enhanced memory coalescing using vectorized loads/stores
// when the input is contiguous along the normalization dimension (stride_C == 1).

template <typename scalar_t>
__global__ void l2_normalize_kernel_coalesced_optimized(
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

    // If the data is contiguous along dim=1, we can enable coalesced accesses
    if (stride_C == 1) {
        // Determine the portion of the vector that can be loaded vectorized
        int aligned_end = (C / 4) * 4;  // number of elements which are a multiple of 4
        int num_vec = aligned_end / 4;  // number of 4-element groups
        
        // Vectorized load & sum
        for (int i = tid; i < num_vec; i += blockDim.x) {
            if constexpr (sizeof(scalar_t) == 4) {
                // For float: use float4 to load 4 floats at a time
                const float4* in_vec = reinterpret_cast<const float4*>(input + base);
                float4 vec = in_vec[i];
                sum += (scalar_t)(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w);
            } else {
                // For double: use double2 to load 2 doubles (16 bytes) at a time
                const double2* in_vec = reinterpret_cast<const double2*>(input + base);
                // Since 4 doubles = 2 double2 loads, adjust the loop accordingly
                // We only vectorize if C is a multiple of 2
                // Compute index in terms of double2 loads
                int num_vec2 = (C / 2);
                if (tid < num_vec2 && i < (num_vec2 / 2)) {
                    const double2* in_vec2 = reinterpret_cast<const double2*>(input + base);
                    double2 vec = in_vec2[i];
                    sum += (scalar_t)(vec.x * vec.x + vec.y * vec.y);
                }
            }
        }

        // Process remaining elements (if C is not divisible by 4 for float or 2 for double)
        for (int i = aligned_end + tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i];
            sum += val * val;
        }
    } else {
        // Fallback for non-contiguous case
        for (int i = tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i * stride_C];
            sum += val * val;
        }
    }

    // Intra-block reduction using warp shuffles and shared memory
    __shared__ scalar_t shared[256];
    int lane = tid % 32;
    int warpId = tid / 32;
    
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0)
        shared[warpId] = sum;
    __syncthreads();

    scalar_t block_sum = (tid < (blockDim.x / 32)) ? shared[lane] : 0;
    if (warpId == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        if (lane == 0)
            shared[0] = block_sum;
    }
    __syncthreads();

    scalar_t norm = sqrt(shared[0]) + 1e-12;
    scalar_t inv_norm = 1.0 / norm;

    // Write back normalized results using coalesced stores if possible
    if (stride_C == 1) {
        int aligned_end = (C / 4) * 4;
        int num_vec = aligned_end / 4;
        if constexpr (sizeof(scalar_t) == 4) {
            // Use vectorized store for floats
            for (int i = tid; i < num_vec; i += blockDim.x) {
                float4* out_vec = reinterpret_cast<float4*>(output + base);
                const float4* in_vec = reinterpret_cast<const float4*>(input + base);
                float4 vec = in_vec[i];
                vec.x *= inv_norm;
                vec.y *= inv_norm;
                vec.z *= inv_norm;
                vec.w *= inv_norm;
                out_vec[i] = vec;
            }
        } else {
            // For doubles, use vectorized store with double2
            int num_vec2 = (C / 2);
            for (int i = tid; i < num_vec2/2; i += blockDim.x) {
                double2* out_vec = reinterpret_cast<double2*>(output + base);
                const double2* in_vec = reinterpret_cast<const double2*>(input + base);
                double2 vec = in_vec[i];
                vec.x *= inv_norm;
                vec.y *= inv_norm;
                out_vec[i] = vec;
            }
        }
        // Process remaining elements
        for (int i = aligned_end + tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i];
            output[base + i] = val * inv_norm;
        }
    } else {
        // Fallback for non-contiguous store
        for (int i = tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i * stride_C];
            output[base + i * stride_C] = val * inv_norm;
        }
    }
}


// C++ interface

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    int C = input.size(1);
    int total_vectors = input.numel() / C;
    int stride_C = input.stride(1);
    int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_coalesced_optimized", ([&] {
        l2_normalize_kernel_coalesced_optimized<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C, total_vectors, stride_C, outer_stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization along dim=1 with coalesced memory access");
}
