#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int d_C;
__constant__ int d_stride_C;
__constant__ int d_outer_stride;
__constant__ int d_elements_per_block;

template <typename scalar_t>
__global__ void l2_normalize_kernel_stage1(
    const scalar_t* input,
    scalar_t* partial_sums,
    const int total_vectors) {

    const int vector_idx = blockIdx.x / ((d_C + d_elements_per_block - 1) / d_elements_per_block);
    const int block_segment = blockIdx.x % ((d_C + d_elements_per_block - 1) / d_elements_per_block);
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * d_outer_stride;
    const int segment_start = block_segment * d_elements_per_block;

    __shared__ scalar_t shared_mem[256];
    scalar_t thread_sum = 0.0;

    #pragma unroll 4
    for (int k = threadIdx.x; k < d_elements_per_block && (segment_start + k) < d_C; k += blockDim.x) {
        const int idx = segment_start + k;
        const scalar_t val = input[base_offset + idx * d_stride_C];
        thread_sum += val * val;
    }

    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        shared_mem[threadIdx.x / 32] = thread_sum;
    }
    __syncthreads();

    if (threadIdx.x < 8) {
        thread_sum = shared_mem[threadIdx.x];
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(&partial_sums[vector_idx], thread_sum);
        }
    }
}

template <typename scalar_t>
__global__ void l2_normalize_kernel_stage2(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* partial_sums,
    const int total_vectors) {

    const int vector_idx = blockIdx.x / ((d_C + d_elements_per_block - 1) / d_elements_per_block);
    const int block_segment = blockIdx.x % ((d_C + d_elements_per_block - 1) / d_elements_per_block);
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * d_outer_stride;
    const int segment_start = block_segment * d_elements_per_block;
    const scalar_t inv_norm = rsqrt(partial_sums[vector_idx] + 1e-12);

    #pragma unroll 4
    for (int k = threadIdx.x; k < d_elements_per_block && (segment_start + k) < d_C; k += blockDim.x) {
        const int idx = segment_start + k;
        output[base_offset + idx * d_stride_C] = input[base_offset + idx * d_stride_C] * inv_norm;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);
    const int elements_per_block = 1024;

    // Copy constants to constant memory
    cudaMemcpyToSymbol(d_C, &C, sizeof(int));
    cudaMemcpyToSymbol(d_stride_C, &stride_C, sizeof(int));
    cudaMemcpyToSymbol(d_outer_stride, &outer_stride, sizeof(int));
    cudaMemcpyToSymbol(d_elements_per_block, &elements_per_block, sizeof(int));

    auto output = torch::empty_like(input);
    auto partial_sums = torch::zeros({total_vectors}, input.options());

    const int threads = 256;
    const int blocks_per_vector = (C + elements_per_block - 1) / elements_per_block;
    const int total_blocks = blocks_per_vector * total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize", [&] {
        l2_normalize_kernel_stage1<scalar_t><<<total_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            total_vectors
        );

        l2_normalize_kernel_stage2<scalar_t><<<total_blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            partial_sums.data_ptr<scalar_t>(),
            total_vectors
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization along dim=1");
}