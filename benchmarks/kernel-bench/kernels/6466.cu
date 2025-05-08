#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Templated kernel with tunable block size
template <typename scalar_t, int BLOCK_SIZE>
__global__ void mean_reduce_kernel_block(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int L,           // reduction dimension length
    int stride,      // stride between reduction elements (inner_size)
    int N) {         // number of output elements

    int out_idx = blockIdx.x;
    if (out_idx >= N) return;

    // Decode output index into outer and inner indices
    int outer_idx = out_idx / stride;
    int inner_idx = out_idx % stride;
    int base_offset = outer_idx * (L * stride) + inner_idx;

    __shared__ scalar_t sdata[BLOCK_SIZE];
    scalar_t sum = static_cast<scalar_t>(0);

    // Each thread processes a subset of the reduction dimension
    for (int i = threadIdx.x; i < L; i += BLOCK_SIZE) {
        sum += __ldg(input + base_offset + i * stride);
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write the computed mean to global memory
    if (threadIdx.x == 0)
        output[out_idx] = sdata[0] / static_cast<scalar_t>(L);
}

// Host function: selects block size based on the reduction dimension size
// and launches the appropriate templated kernel

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    // Get the dimensions
    std::vector<int64_t> sizes = input.sizes().vec();
    int64_t L = sizes[dim];

    // Compute outer_size (product of dims before 'dim') and inner_size (after 'dim')
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }
    
    // Total number of output elements
    int64_t N = outer_size * inner_size;
    int stride = inner_size;  // reduction elements are separated by inner_size

    // Remove the reduced dimension from shape for the output
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty({N}, input.options());

    // Experiment with block sizes based on the size of the reduction dimension L
    int block_size_candidate = 256; // default
    
    if (L <= 32) {
        block_size_candidate = 32;
    } else if (L <= 64) {
        block_size_candidate = 64;
    } else if (L <= 128) {
        block_size_candidate = 128;
    } else if (L <= 256) {
        block_size_candidate = 256;
    } else if (L <= 512) {
        block_size_candidate = 512;
    } else {
        block_size_candidate = 256; // for very large L, 256 tends to balance occupancy and resource usage
    }

    const int threads = block_size_candidate;
    const int blocks = static_cast<int>(N);  // one block per output element

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        if (block_size_candidate == 32) {
            mean_reduce_kernel_block<scalar_t, 32><<<blocks, 32>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<int>(L),
                stride,
                static_cast<int>(N)
            );
        } else if (block_size_candidate == 64) {
            mean_reduce_kernel_block<scalar_t, 64><<<blocks, 64>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<int>(L),
                stride,
                static_cast<int>(N)
            );
        } else if (block_size_candidate == 128) {
            mean_reduce_kernel_block<scalar_t, 128><<<blocks, 128>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<int>(L),
                stride,
                static_cast<int>(N)
            );
        } else if (block_size_candidate == 256) {
            mean_reduce_kernel_block<scalar_t, 256><<<blocks, 256>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<int>(L),
                stride,
                static_cast<int>(N)
            );
        } else if (block_size_candidate == 512) {
            mean_reduce_kernel_block<scalar_t, 512><<<blocks, 512>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<int>(L),
                stride,
                static_cast<int>(N)
            );
        }
    }));

    output = output.view(sizes);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean Reduction with Tunable Block Size (CUDA)");
}
