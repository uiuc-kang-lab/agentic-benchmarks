#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void l2_normalize_kernel_stage1(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ partial_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int elements_per_block) {

    const int vector_idx = blockIdx.x / ((C + elements_per_block - 1) / elements_per_block);
    const int block_segment = blockIdx.x % ((C + elements_per_block - 1) / elements_per_block);
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;
    const int segment_start = block_segment * elements_per_block;

    // Pad shared memory to avoid bank conflicts
    __shared__ scalar_t shared_mem[32 * 33];  // 33 instead of 32 to avoid bank conflicts
    scalar_t thread_sum = 0.0;

    // Align memory access to 128-bit boundaries where possible
    const scalar_t* aligned_input = input + base_offset;
    
    // Use vectorized loads for better memory throughput
    #pragma unroll 8
    for (int k = threadIdx.x; k < elements_per_block && (segment_start + k) < C; k += blockDim.x) {
        const int idx = segment_start + k;
        const scalar_t val = __ldg(&aligned_input[idx * stride_C]);
        thread_sum += val * val;
    }

    // Store to padded shared memory to avoid bank conflicts
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    
    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Write reduced warp results to shared memory with padding
    if (lane_id == 0) {
        shared_mem[warp_id * 33] = thread_sum;
    }
    __syncthreads();

    // Final reduction across warps
    if (threadIdx.x < 32) {
        thread_sum = (threadIdx.x < (blockDim.x / 32)) ? shared_mem[threadIdx.x * 33] : 0.0;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        if (threadIdx.x == 0) {
            atomicAdd(&partial_sums[vector_idx], thread_sum);
        }
    }
}

template <typename scalar_t>
__global__ void l2_normalize_kernel_stage2(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ partial_sums,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int elements_per_block) {

    const int vector_idx = blockIdx.x / ((C + elements_per_block - 1) / elements_per_block);
    const int block_segment = blockIdx.x % ((C + elements_per_block - 1) / elements_per_block);
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;
    const int segment_start = block_segment * elements_per_block;
    
    // Use __ldg for read-only access to partial sums
    const scalar_t inv_norm = rsqrt(__ldg(&partial_sums[vector_idx]) + 1e-12);

    // Align memory access and use vectorized loads where possible
    const scalar_t* aligned_input = input + base_offset;
    scalar_t* aligned_output = output + base_offset;

    #pragma unroll 4
    for (int k = threadIdx.x; k < elements_per_block && (segment_start + k) < C; k += blockDim.x) {
        const int idx = segment_start + k;
        const scalar_t val = __ldg(&aligned_input[idx * stride_C]);
        aligned_output[idx * stride_C] = val * inv_norm;
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
    auto partial_sums = torch::zeros({total_vectors}, input.options());

    const int threads = 512;
    const int elements_per_block = 2048;
    const int blocks_per_vector = (C + elements_per_block - 1) / elements_per_block;
    const int total_blocks = blocks_per_vector * total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize", [&] {
        l2_normalize_kernel_stage1<scalar_t><<<total_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride,
            elements_per_block
        );

        l2_normalize_kernel_stage2<scalar_t><<<total_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride,
            elements_per_block
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization along dim=1");
}