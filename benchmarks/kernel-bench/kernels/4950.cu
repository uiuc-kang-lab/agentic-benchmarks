#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to perform L2 normalization using grid-stride loops to balance workload
template <typename scalar_t>
__global__ void l2_normalize_kernel_grid_stride(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    // Process multiple vectors per block using a grid-stride loop
    for (int vector_idx = blockIdx.x; vector_idx < total_vectors; vector_idx += gridDim.x) {
        int base_offset = vector_idx * outer_stride;
        scalar_t sum = 0.0;

        // Each thread computes partial sum of squares for this vector
        for (int k = threadIdx.x; k < C; k += blockDim.x) {
            scalar_t val = input[base_offset + k * stride_C];
            sum += val * val;
        }

        // Perform warp-level reduction using shfl_down_sync
        int lane = threadIdx.x & 31;
        int warp_id = threadIdx.x >> 5; // divide by 32
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Use shared memory to accumulate results across warps
        // We assume a maximum of 256 threads per block -> 256/32 = 8 warps
        __shared__ scalar_t shared[8];
        if (lane == 0) {
            shared[warp_id] = sum;
        }
        __syncthreads();

        // First warp loads the partial sums and reduces them
        if (threadIdx.x < blockDim.x / 32) {
            sum = shared[lane];
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (threadIdx.x == 0) {
                shared[0] = sum;
            }
        }
        __syncthreads();

        // Total sum for this vector
        scalar_t total_sum = shared[0];
        scalar_t inv_norm = 1.0 / (sqrt(total_sum) + 1e-12);

        // Normalize: each thread processes a subset of elements
        for (int k = threadIdx.x; k < C; k += blockDim.x) {
            output[base_offset + k * stride_C] = input[base_offset + k * stride_C] * inv_norm;
        }
        __syncthreads(); // Ensure all threads finish before next iteration
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

    // Choose a reasonable number of blocks to balance work across SMs
    int blocks = total_vectors < 1024 ? total_vectors : 1024;
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize_grid_stride", ([&] {
        l2_normalize_kernel_grid_stride<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &forward, "L2 normalization along dim=1 using grid-stride loop");
}
