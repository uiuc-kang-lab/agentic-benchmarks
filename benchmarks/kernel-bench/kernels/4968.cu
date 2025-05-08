#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void l2norm_coalesced_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {
    
    // Align to 128-byte boundaries for optimal memory coalescing on H100
    __shared__ scalar_t shared[32][33]; // 33 for bank conflict avoidance
    
    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;
    
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    const int base = vector_idx * outer_stride;
    
    scalar_t sum = 0;
    
    if (stride_C == 1) {
        // Coalesced memory access path
        using Vec4 = typename std::conditional<sizeof(scalar_t) == 4, float4, double2>::type;
        constexpr int vec_size = sizeof(Vec4) / sizeof(scalar_t);
        
        // Each thread processes consecutive elements
        const Vec4* vec_input = reinterpret_cast<const Vec4*>(input + base);
        const int vec_elements = C / vec_size;
        const int elements_per_thread = (vec_elements + blockDim.x - 1) / blockDim.x;
        
        #pragma unroll 4
        for (int i = 0; i < elements_per_thread; i++) {
            const int idx = tid + i * blockDim.x;
            if (idx < vec_elements) {
                Vec4 v = vec_input[idx];
                if constexpr (sizeof(scalar_t) == 4) {
                    sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                } else {
                    sum += v.x * v.x + v.y * v.y;
                }
            }
        }
        
        // Handle remaining elements
        for (int i = C - (C % vec_size) + tid; i < C; i += blockDim.x) {
            scalar_t val = input[base + i];
            sum += val * val;
        }
    } else {
        // Non-contiguous case: ensure coalesced access within warps
        const int warps = blockDim.x / 32;
        const int elements_per_warp = (C + warps - 1) / warps;
        const int warp_start = wid * elements_per_warp;
        
        #pragma unroll 4
        for (int i = 0; i < (elements_per_warp + 31) / 32; i++) {
            const int idx = warp_start + i * 32 + lane;
            if (idx < C) {
                scalar_t val = input[base + idx * stride_C];
                sum += val * val;
            }
        }
    }
    
    // Warp-level reduction with sequential addressing
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Store warp results in shared memory with padding to avoid bank conflicts
    if (lane == 0) {
        shared[wid][0] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < (blockDim.x / 32)) {
        sum = shared[tid][0];
        
        #pragma unroll
        for (int offset = (blockDim.x / 64); offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (tid == 0) {
            shared[0][0] = sqrt(sum) + scalar_t(1e-12);
        }
    }
    __syncthreads();
    
    const scalar_t inv_norm = scalar_t(1.0) / shared[0][0];
    
    // Normalized write-back with coalesced access
    if (stride_C == 1) {
        using Vec4 = typename std::conditional<sizeof(scalar_t) == 4, float4, double2>::type;
        constexpr int vec_size = sizeof(Vec4) / sizeof(scalar_t);
        
        Vec4* vec_output = reinterpret_cast<Vec4*>(output + base);
        const Vec4* vec_input = reinterpret_cast<const Vec4*>(input + base);
        const int vec_elements = C / vec_size;
        const int elements_per_thread = (vec_elements + blockDim.x - 1) / blockDim.x;
        
        #pragma unroll 4
        for (int i = 0; i < elements_per_thread; i++) {
            const int idx = tid + i * blockDim.x;
            if (idx < vec_elements) {
                Vec4 v = vec_input[idx];
                if constexpr (sizeof(scalar_t) == 4) {
                    v.x *= inv_norm;
                    v.y *= inv_norm;
                    v.z *= inv_norm;
                    v.w *= inv_norm;
                } else {
                    v.x *= inv_norm;
                    v.y *= inv_norm;
                }
                vec_output[idx] = v;
            }
        }
        
        // Handle remaining elements
        for (int i = C - (C % vec_size) + tid; i < C; i += blockDim.x) {
            output[base + i] = input[base + i] * inv_norm;
        }
    } else {
        // Non-contiguous case: maintain coalesced access within warps
        const int warps = blockDim.x / 32;
        const int elements_per_warp = (C + warps - 1) / warps;
        const int warp_start = wid * elements_per_warp;
        
        #pragma unroll 4
        for (int i = 0; i < (elements_per_warp + 31) / 32; i++) {
            const int idx = warp_start + i * 32 + lane;
            if (idx < C) {
                output[base + idx * stride_C] = input[base + idx * stride_C] * inv_norm;
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
    
    const int threads = 256; // Multiple of warp size for optimal occupancy
    const int blocks = total_vectors;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_coalesced", ([&] {
        l2norm_coalesced_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C, total_vectors, stride_C, outer_stride);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with fully coalesced memory access");
}