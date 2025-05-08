#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Modular device function to load an element with caching
template <typename scalar_t>
__device__ inline scalar_t load_element(const scalar_t* ptr) {
    return __ldg(ptr);
}

// Device function to compute a partial sum over the reduction dimension
// Each thread processes indices starting at its tid and stepping by blockDim.x
template <typename scalar_t>
__device__ inline scalar_t compute_partial_sum(const scalar_t* data, int L, int stride, int tid, int blockSize) {
    scalar_t sum = static_cast<scalar_t>(0);
    for (int i = tid; i < L; i += blockSize) {
        sum += load_element(data + i * stride);
    }
    return sum;
}

// Device function to perform block-level reduction using shared memory
// This function assumes that sdata[tid] holds the partial sum for each thread
template <typename scalar_t>
__device__ inline scalar_t reduce_block(volatile scalar_t* sdata, int blockSize) {
    int tid = threadIdx.x;
    if (blockSize >= 512) { 
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } 
        __syncthreads();
    }
    if (blockSize >= 256) { 
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } 
        __syncthreads();
    }
    if (blockSize >= 128) { 
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } 
        __syncthreads();
    }
    if (tid < 32) {
        volatile scalar_t* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 32];
        vsdata[tid] += vsdata[tid + 16];
        vsdata[tid] += vsdata[tid + 8];
        vsdata[tid] += vsdata[tid + 4];
        vsdata[tid] += vsdata[tid + 2];
        vsdata[tid] += vsdata[tid + 1];
    }
    return sdata[0];
}

// Main kernel that computes the mean reduction over a specified dimension
// The kernel is modular: it uses compute_partial_sum to accumulate sums and reduce_block for intra-block reduction.
// Each block computes one output element corresponding to a slice along the reduction dimension.
template <typename scalar_t>
__global__ void modular_mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int L,      // Length along the reduction dimension
    int stride, // Stride between consecutive elements in the reduction dimension
    int N       // Total number of output elements
) {
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    int out_idx = blockIdx.x; // Each block computes one output element
    if (out_idx >= N) return;

    // Decode the flat output index into outer and inner indices
    int outer_idx = out_idx / stride;
    int inner_idx = out_idx % stride;
    int base_offset = outer_idx * (L * stride) + inner_idx;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Compute this thread's partial sum over the reduction dimension
    scalar_t partial = compute_partial_sum<scalar_t>(input + base_offset, L, stride, tid, blockSize);
    sdata[tid] = partial;
    __syncthreads();

    // Reduce the partial sums stored in shared memory to a single sum
    if (tid == 0) {
        scalar_t block_sum = reduce_block<scalar_t>(sdata, blockSize);
        output[out_idx] = block_sum / static_cast<scalar_t>(L);
    }
}

// Host function to set up tensor dimensions and launch the modular kernel
// This function removes the reduced dimension from the output tensor shape
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    // Get input dimensions and compute sizes for outer, reduction (L), and inner parts
    std::vector<int64_t> sizes = input.sizes().vec();
    int64_t L = sizes[dim];

    int64_t outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= sizes[i];
    }
    int64_t inner = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner *= sizes[i];
    }

    int64_t N = outer * inner; // Total number of output elements
    int stride = inner; // Stride for accessing elements along the reduction dimension

    // Prepare output tensor shape by removing the reduced dimension
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty({N}, input.options());

    // Set kernel launch configuration
    int blockSize = 256;
    int sharedMemSize = blockSize * sizeof(float); // this works for float; for double, it uses sizeof(scalar_t) via dispatch

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        modular_mean_reduce_kernel<scalar_t><<<
            static_cast<int>(N),
            blockSize,
            blockSize * sizeof(scalar_t)
        >>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<int>(L),
            stride,
            static_cast<int>(N)
        );
    }));

    return output.view(sizes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Modular Mean Reduction (CUDA)");
}
