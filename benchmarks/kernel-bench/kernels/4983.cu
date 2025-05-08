#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// This kernel assigns one thread block per vector. By doing so, reduction across the vector
// is handled entirely within the block using shared memory and warp-level primitives, eliminating
// any need for global atomic operations. Each thread processes a subset of the vector using a
// grid-stride loop and then the block collectively computes the norm, which is used to normalize
// the elements.

template <typename scalar_t>
__global__ void l2_norm_fused_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int outer_stride,
    const int stride_C) {

    // Each block processes one vector
    int vector_idx = blockIdx.x;
    int total_vectors = gridDim.x;  // one block per vector
    if (vector_idx >= total_vectors) return;

    int base = vector_idx * outer_stride;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    // Compute partial sum of squares over the vector using a grid-stride loop
    scalar_t sum = 0;
    if (stride_C == 1) {
        // Contiguous case: elements are laid out consecutively
        for (int i = tid; i < C; i += blockSize) {
            scalar_t val = input[base + i];
            sum += val * val;
        }
    } else {
        // Non-contiguous: jump by stride_C
        for (int i = tid; i < C; i += blockSize) {
            scalar_t val = input[base + i * stride_C];
            sum += val * val;
        }
    }

    // Warp-level reduction: reduce sum within each warp using shuffle instructions
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Use shared memory to combine results across warps
    __shared__ scalar_t shared[32];  // maximum number of warps per block (256/32 = 8, allocate 32 for safety)
    int lane = tid & 31;
    int warpId = tid >> 5;
    if (lane == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();

    // Final reduction from shared memory using the first warp
    if (tid < blockSize / 32) {
        sum = shared[lane];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            shared[0] = sum;
        }
    }
    __syncthreads();

    // Compute the normalization factor
    scalar_t norm = sqrt(shared[0]) + static_cast<scalar_t>(1e-12);
    scalar_t inv_norm = static_cast<scalar_t>(1.0) / norm;

    // Normalize the vector using the same grid-stride loop
    if (stride_C == 1) {
        for (int i = tid; i < C; i += blockSize) {
            output[base + i] = input[base + i] * inv_norm;
        }
    } else {
        for (int i = tid; i < C; i += blockSize) {
            output[base + i * stride_C] = input[base + i * stride_C] * inv_norm;
        }
    }
}


// Host forward function
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    // Assuming normalization is along dim=1
    const int C = input.size(1);
    const int total_vectors = input.numel() / C;  // works for contiguous 2D tensors and similar layouts
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    // Launch one block per vector; using 256 threads per block
    const int threads = 256;
    dim3 blocks(total_vectors);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_fused_kernel", ([&] {
        l2_norm_fused_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C, outer_stride, stride_C
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization fused kernel with minimized atomic operations");
}
