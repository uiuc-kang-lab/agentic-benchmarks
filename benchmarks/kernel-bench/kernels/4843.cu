#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define UNROLL_FACTOR 4

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void l1_norm_forward_kernel_unrolled(const float* __restrict__ x,
                                              float* __restrict__ out,
                                              const int N,
                                              const int D) {
    const int row = blockIdx.x;
    float thread_sum = 0.0f;
    
    if (D >= 16) {  // Use vectorized loads for larger dimensions
        const int vec_elements = D / 4;
        const float4* x_vec = reinterpret_cast<const float4*>(x + row * D);
        
        #pragma unroll UNROLL_FACTOR
        for (int base = threadIdx.x; base < vec_elements; base += blockDim.x) {
            float4 data = __ldg(&x_vec[base]);
            thread_sum += fabsf(data.x) + fabsf(data.y) + fabsf(data.z) + fabsf(data.w);
        }
        
        // Handle remaining elements
        #pragma unroll
        for (int i = D - (D % 4) + threadIdx.x; i < D; i += blockDim.x) {
            thread_sum += fabsf(__ldg(&x[row * D + i]));
        }
    } else {
        // For small D, use scalar loads with manual unrolling
        #pragma unroll 4
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            thread_sum += fabsf(__ldg(&x[row * D + i]));
        }
    }

    // Warp-level reduction using shuffle
    thread_sum = warp_reduce_sum(thread_sum);

    // Block-level reduction using shared memory
    extern __shared__ float sdata[];
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    if (lane_id == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction and normalization constant computation
    if (threadIdx.x == 0) {
        float final_sum = 0.0f;
        const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        
        #pragma unroll
        for (int i = 0; i < num_warps; ++i) {
            final_sum += sdata[i];
        }
        sdata[0] = (final_sum > 0.0f) ? final_sum : 1e-12f;
    }
    __syncthreads();
    
    const float norm_factor = sdata[0];

    // Normalize using vectorized operations when possible
    if (D >= 16) {
        const int vec_elements = D / 4;
        const float4* x_vec = reinterpret_cast<const float4*>(x + row * D);
        float4* out_vec = reinterpret_cast<float4*>(out + row * D);
        
        #pragma unroll UNROLL_FACTOR
        for (int base = threadIdx.x; base < vec_elements; base += blockDim.x) {
            float4 data = __ldg(&x_vec[base]);
            data.x /= norm_factor;
            data.y /= norm_factor;
            data.z /= norm_factor;
            data.w /= norm_factor;
            out_vec[base] = data;
        }

        // Handle remaining elements
        #pragma unroll
        for (int i = D - (D % 4) + threadIdx.x; i < D; i += blockDim.x) {
            out[row * D + i] = __ldg(&x[row * D + i]) / norm_factor;
        }
    } else {
        // For small D, use scalar operations with manual unrolling
        #pragma unroll 4
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            out[row * D + i] = __ldg(&x[row * D + i]) / norm_factor;
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
    x = x.contiguous();

    const int N = x.size(0);
    const int D = x.size(1);
    auto out = torch::empty_like(x);

    // Optimize thread count based on dimension size
    const int thread_count = (D >= 512) ? 512 : ((D >= 256) ? 256 : ((D >= 128) ? 128 : 64));
    const int warps = (thread_count + WARP_SIZE - 1) / WARP_SIZE;
    const int shmem_size = warps * sizeof(float);

    l1_norm_forward_kernel_unrolled<<<N, thread_count, shmem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward with unrolled loops (CUDA)");
}