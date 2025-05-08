#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void l2_normalize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int wid = tid / 32;
    
    // Early exit for entire warps
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;
    
    // Calculate number of elements per warp to process
    const int vector_size = 4;
    const int elements_per_warp = ((C + 31) / 32) * 32; // Round up to nearest warp size
    const int warp_offset = (tid / 32) * elements_per_warp;
    
    scalar_t sum = 0.0;
    
    // Process aligned chunks with float4
    const int aligned_elements = (C / (vector_size * 32)) * (vector_size * 32);
    float4 vec_val;
    
    #pragma unroll
    for (int k = warp_offset + lane * vector_size; k < aligned_elements; k += elements_per_warp) {
        vec_val = *reinterpret_cast<const float4*>(&input[base_offset + k * stride_C]);
        sum += vec_val.x * vec_val.x + vec_val.y * vec_val.y + 
               vec_val.z * vec_val.z + vec_val.w * vec_val.w;
    }
    
    // Process remaining elements within the same warp
    for (int k = aligned_elements + warp_offset + lane; k < C; k += 32) {
        const scalar_t val = input[base_offset + k * stride_C];
        sum += val * val;
    }

    // Warp-synchronized reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Only first thread in each warp writes to shared memory
    __shared__ scalar_t shared_sum[8]; // Assuming max 8 warps per block
    if (lane == 0) {
        shared_sum[wid] = sum;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (wid == 0) {
        sum = (lane < (blockDim.x / 32)) ? shared_sum[lane] : 0.0;
        
        #pragma unroll
        for (int offset = (blockDim.x / 64); offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane == 0) {
            shared_sum[0] = sum;
        }
    }
    __syncthreads();

    const scalar_t inv_norm = 1.0 / (sqrt(shared_sum[0]) + 1e-12);

    // Normalize with vectorized loads/stores
    #pragma unroll
    for (int k = warp_offset + lane * vector_size; k < aligned_elements; k += elements_per_warp) {
        vec_val = *reinterpret_cast<const float4*>(&input[base_offset + k * stride_C]);
        vec_val.x *= inv_norm;
        vec_val.y *= inv_norm;
        vec_val.z *= inv_norm;
        vec_val.w *= inv_norm;
        *reinterpret_cast<float4*>(&output[base_offset + k * stride_C]) = vec_val;
    }

    // Process remaining elements with uniform warp execution
    for (int k = aligned_elements + warp_offset + lane; k < C; k += 32) {
        output[base_offset + k * stride_C] = input[base_offset + k * stride_C] * inv_norm;
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
        l2_normalize_kernel<scalar_t><<<blocks, threads>>>(
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