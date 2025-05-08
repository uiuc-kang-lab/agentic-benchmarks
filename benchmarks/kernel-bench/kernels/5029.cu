#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Helper function to calculate grid size
inline int calculate_grid_size(int total_threads, int block_size) {
    return (total_threads + block_size - 1) / block_size;
}

template <typename scalar_t>
__global__ void improved_l2_normalize_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    const int vector_idx = blockIdx.x;
    if (vector_idx >= total_vectors) return;

    const int base_offset = vector_idx * outer_stride;

    scalar_t sum = 0.0;

    // Compute sum of squares in a balanced manner
    for (int k = threadIdx.x; k < C; k += blockDim.x) {
        const scalar_t val = input[base_offset + k * stride_C];
        sum += val * val;
    }

    // Block-wise reduction using warp-level primitives first
    __shared__ scalar_t cache[256];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    
    // Warp-level reduction first
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write reduced warp results to shared memory
    if (lane == 0) {
        cache[wid] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < (blockDim.x / 32)) {
        sum = cache[tid];
    } else {
        sum = 0;
    }
    __syncthreads();
    
    if (tid < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    
    // Broadcast final sum to all threads
    if (tid == 0) {
        cache[0] = sum;
    }
    __syncthreads();
    sum = cache[0];
    __syncthreads();

    const scalar_t inv_norm = 1.0 / (sqrt(sum) + 1e-12);

    // Each thread normalizes its allocated workload
    for (int k = threadIdx.x; k < C; k += blockDim.x) {
        output[base_offset + k * stride_C] = input[base_offset + k * stride_C] * inv_norm;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0); // Simplified assumption for contiguous tensors

    auto output = torch::empty_like(input);

    // Setting block and grid sizes
    const int threads = 256;
    const int blocks = calculate_grid_size(total_vectors, threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize", [&] {
        improved_l2_normalize_kernel<scalar_t><<<blocks, threads>>>(
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