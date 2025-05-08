#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void l2norm_strided_kernel(
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
    const int stride = blockDim.x;
    
    // Use shared memory for partial sums
    __shared__ scalar_t shared_mem[256];
    scalar_t thread_sum = 0;

    if (stride_C == 1) {
        // Vectorized load path for contiguous data
        const int vec_size = sizeof(scalar_t) == 4 ? 4 : 2;
        const int aligned_C = (C / vec_size) * vec_size;
        
        // Process vectorized loads with stride
        if constexpr (sizeof(scalar_t) == 4) {
            const float4* in_vec = reinterpret_cast<const float4*>(input + base);
            const int num_vectors = aligned_C / 4;
            
            // Each thread processes multiple vectors in strided fashion
            for (int i = tid; i < num_vectors; i += stride) {
                float4 v = in_vec[i];
                thread_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        } else {
            const double2* in_vec = reinterpret_cast<const double2*>(input + base);
            const int num_vectors = aligned_C / 2;
            
            for (int i = tid; i < num_vectors; i += stride) {
                double2 v = in_vec[i];
                thread_sum += v.x * v.x + v.y * v.y;
            }
        }

        // Handle remaining elements
        for (int i = aligned_C + tid; i < C; i += stride) {
            scalar_t val = input[base + i];
            thread_sum += val * val;
        }
    } else {
        // Non-contiguous data handling with stride loops
        for (int i = tid; i < C; i += stride) {
            scalar_t val = input[base + i * stride_C];
            thread_sum += val * val;
        }
    }

    // Store partial sum
    shared_mem[tid] = thread_sum;
    __syncthreads();

    // Reduction within block using stride loops
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < 32) {
        // Volatile pointer for warp-synchronous programming
        volatile scalar_t* smem = shared_mem;
        if (blockDim.x > 64) smem[tid] += smem[tid + 32];
        if (blockDim.x > 32) smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    // Compute normalization factor
    if (tid == 0) {
        shared_mem[0] = rsqrt(shared_mem[0] + 1e-12);
    }
    __syncthreads();

    const scalar_t inv_norm = shared_mem[0];

    // Normalize using stride loops
    if (stride_C == 1) {
        // Vectorized store path for contiguous data
        const int vec_size = sizeof(scalar_t) == 4 ? 4 : 2;
        const int aligned_C = (C / vec_size) * vec_size;

        if constexpr (sizeof(scalar_t) == 4) {
            float4* out_vec = reinterpret_cast<float4*>(output + base);
            const float4* in_vec = reinterpret_cast<const float4*>(input + base);
            const int num_vectors = aligned_C / 4;

            for (int i = tid; i < num_vectors; i += stride) {
                float4 v = in_vec[i];
                v.x *= inv_norm;
                v.y *= inv_norm;
                v.z *= inv_norm;
                v.w *= inv_norm;
                out_vec[i] = v;
            }
        } else {
            double2* out_vec = reinterpret_cast<double2*>(output + base);
            const double2* in_vec = reinterpret_cast<const double2*>(input + base);
            const int num_vectors = aligned_C / 2;

            for (int i = tid; i < num_vectors; i += stride) {
                double2 v = in_vec[i];
                v.x *= inv_norm;
                v.y *= inv_norm;
                out_vec[i] = v;
            }
        }

        // Handle remaining elements with stride
        for (int i = aligned_C + tid; i < C; i += stride) {
            output[base + i] = input[base + i] * inv_norm;
        }
    } else {
        // Non-contiguous data handling with stride loops
        for (int i = tid; i < C; i += stride) {
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

    // Choose optimal thread block size based on C
    const int threads = 256;  // Optimal for H100
    const dim3 blocks(total_vectors);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_strided", ([&] {
        l2norm_strided_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &forward, "L2 normalization with stride optimization");
}
