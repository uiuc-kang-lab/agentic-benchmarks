#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void l2_normalize_kernel_coalesced(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int tid = threadIdx.x;
    const int base_offset = vector_idx * outer_stride;
    
    // Use float4 for coalesced memory access when possible
    const int vec_size = sizeof(float4) / sizeof(scalar_t);
    const int C_aligned = C / vec_size * vec_size;
    
    scalar_t sum = 0.0;
    
    // Process aligned elements using float4
    if (tid < (C_aligned / vec_size)) {
        const float4* input4 = reinterpret_cast<const float4*>(input + base_offset);
        float4 val4 = input4[tid];
        sum += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
    }
    
    // Process remaining elements
    for (int k = C_aligned + tid; k < C; k += blockDim.x) {
        const scalar_t val = input[base_offset + k];
        sum += val * val;
    }

    // Block-wide reduction
    __shared__ scalar_t shared_sum[256];
    const int lane = tid % 32;
    const int wid = tid / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        shared_sum[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        sum = (lane < (blockDim.x / 32)) ? shared_sum[lane] : 0.0;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane == 0) {
            shared_sum[0] = sum;
        }
    }
    __syncthreads();

    const scalar_t inv_norm = 1.0 / (sqrt(shared_sum[0]) + 1e-12);

    // Normalize using vectorized operations when possible
    if (tid < (C_aligned / vec_size)) {
        const float4* input4 = reinterpret_cast<const float4*>(input + base_offset);
        float4* output4 = reinterpret_cast<float4*>(output + base_offset);
        float4 val4 = input4[tid];
        
        val4.x *= inv_norm;
        val4.y *= inv_norm;
        val4.z *= inv_norm;
        val4.w *= inv_norm;
        
        output4[tid] = val4;
    }
    
    // Process remaining elements
    for (int k = C_aligned + tid; k < C; k += blockDim.x) {
        output[base_offset + k] = input[base_offset + k] * inv_norm;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize", [&] {
        l2_normalize_kernel_coalesced<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization along dim=1");
}