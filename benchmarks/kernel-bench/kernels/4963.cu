#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel assumes that the input tensor is contiguous along the normalization dimension (stride_C == 1) to guarantee that
// global memory accesses are coalesced. It uses vectorized loads/stores (float4 for float and double2 for double) to ensure that
// consecutive threads in a warp access consecutive memory locations. For non-contiguous cases (stride_C != 1), it falls back to a safe loop.

template <typename scalar_t>
__global__ void l2_norm_coalesced_aligned_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    // Each block processes one vector
    int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;
    int base_offset = vector_idx * outer_stride;
    int tid = threadIdx.x;
    scalar_t sum = 0;

    // Use coalesced, vectorized loads if the data is contiguous
    if (stride_C == 1) {
        if constexpr (sizeof(scalar_t) == 4) {
            int vec_length = (C / 4) * 4;  // number of elements covered by vectorized loads
            const float4* input_vec = reinterpret_cast<const float4*>(input + base_offset);
            int num_vec = vec_length / 4;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                float4 v = input_vec[i];
                sum += (scalar_t)(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
            }
            // Handle remaining elements
            for (int i = vec_length + tid; i < C; i += blockDim.x) {
                float val = input[base_offset + i];
                sum += val * val;
            }
        } else {
            // For double precision, use double2
            int vec_length = (C / 2) * 2;
            const double2* input_vec = reinterpret_cast<const double2*>(input + base_offset);
            int num_vec = vec_length / 2;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                double2 v = input_vec[i];
                sum += (scalar_t)(v.x * v.x + v.y * v.y);
            }
            // Handle remaining elements
            for (int i = vec_length + tid; i < C; i += blockDim.x) {
                double val = input[base_offset + i];
                sum += val * val;
            }
        }
    } else {
        // Fallback for non-contiguous case
        for (int i = tid; i < C; i += blockDim.x) {
            scalar_t x = input[base_offset + i * stride_C];
            sum += x * x;
        }
    }

    // Intra-warp reduction using warp shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Use shared memory to accumulate the results of each warp
    __shared__ scalar_t shared[32];  // Enough space for up to 32 warps
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction from each warp's result
    if (warp_id == 0) {
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    // The first thread computes the normalization factor and broadcasts it
    __shared__ scalar_t inv_norm;
    if (tid == 0) {
        scalar_t norm = sqrt(sum) + (scalar_t)1e-12;
        inv_norm = (scalar_t)1.0 / norm;
    }
    __syncthreads();

    // Write back normalized results with coalesced, vectorized stores if possible
    if (stride_C == 1) {
        if constexpr (sizeof(scalar_t) == 4) {
            int vec_length = (C / 4) * 4;
            float4* output_vec = reinterpret_cast<float4*>(output + base_offset);
            const float4* input_vec = reinterpret_cast<const float4*>(input + base_offset);
            int num_vec = vec_length / 4;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                float4 v = input_vec[i];
                v.x *= inv_norm;
                v.y *= inv_norm;
                v.z *= inv_norm;
                v.w *= inv_norm;
                output_vec[i] = v;
            }
            // Process remaining elements
            for (int i = vec_length + tid; i < C; i += blockDim.x) {
                output[base_offset + i] = input[base_offset + i] * inv_norm;
            }
        } else {
            int vec_length = (C / 2) * 2;
            double2* output_vec = reinterpret_cast<double2*>(output + base_offset);
            const double2* input_vec = reinterpret_cast<const double2*>(input + base_offset);
            int num_vec = vec_length / 2;
            for (int i = tid; i < num_vec; i += blockDim.x) {
                double2 v = input_vec[i];
                v.x *= inv_norm;
                v.y *= inv_norm;
                output_vec[i] = v;
            }
            for (int i = vec_length + tid; i < C; i += blockDim.x) {
                output[base_offset + i] = input[base_offset + i] * inv_norm;
            }
        }
    } else {
        // Fallback for non-contiguous store
        for (int i = tid; i < C; i += blockDim.x) {
            output[base_offset + i * stride_C] = input[base_offset + i * stride_C] * inv_norm;
        }
    }
}

// Forward function that launches one block per vector
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor.");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D.");
    
    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    dim3 blocks(total_vectors);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_coalesced_aligned", ([&] {
        l2_norm_coalesced_aligned_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C, total_vectors, stride_C, outer_stride);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with aligned/coalesced memory accesses");
}
