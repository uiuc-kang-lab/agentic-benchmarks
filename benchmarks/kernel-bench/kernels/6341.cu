#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block tile dimensions
constexpr int TX = 128; // number of threads processing inner dimension
constexpr int TY = 8;   // number of threads parallelizing reduction

// Modular device function: warp-level reduction
template <typename scalar_t>
__device__ inline scalar_t warpReduceSum(scalar_t val) {
    const int warpSize = 32;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Kernel: Each block processes one outer index and a tile of the inner dimension.
// Threads cooperatively load portions of the reduction dimension into shared memory and reduce them.
template <typename scalar_t>
__global__ void optimized_shared_warp_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    int outer_idx = blockIdx.x;
    int inner_tile_start = blockIdx.y * blockDim.x;  // blockDim.x == TX

    int tx = threadIdx.x; // index within the inner tile
    int ty = threadIdx.y; // used for parallelizing the reduction

    int inner_idx = inner_tile_start + tx;

    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    scalar_t partial_sum = 0;
    if (inner_idx < inner_size) {
        for (int r = ty; r < reduce_size; r += blockDim.y) {
            int64_t idx = outer_idx * (reduce_size * inner_size) + r * inner_size + inner_idx;
            partial_sum += input[idx];
        }
    }

    int index = ty * blockDim.x + tx;
    sdata[index] = partial_sum;
    __syncthreads();

    // Reduce within shared memory using warp-level reduction
    if (ty == 0) {
        partial_sum = warpReduceSum(sdata[tx]);
        if (tx % 32 == 0) {
            sdata[tx] = partial_sum;
        }
    }
    __syncthreads();

    if (ty == 0 && tx < 32) {
        scalar_t final_sum = warpReduceSum(sdata[tx]);
        if (tx == 0 && inner_idx < inner_size) {
            int64_t o_idx = outer_idx * inner_size + inner_idx;
            output[o_idx] = final_sum;
        }
    }
}

// CUDA wrapper function
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) {
        dim += input.dim();
    }

    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    dim3 threads(TX, TY);
    dim3 blocks(outer_size, (inner_size + TX - 1) / TX);

    size_t shared_mem_size = TX * TY * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        optimized_shared_warp_reduce_kernel<scalar_t><<<blocks, threads, TX * TY * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Optimized shared and warp-level sum reduction (CUDA)");
}