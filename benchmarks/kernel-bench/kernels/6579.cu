#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel ensures memory coalescing by aligning global memory accesses
// so threads in a warp read/write consecutive memory locations.

template <typename scalar_t>
__global__ void coalesced_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    extern __shared__ char sdata[];
    scalar_t* shmax = reinterpret_cast<scalar_t*>(sdata);

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int out_idx = blockIdx.x * block_size + tid;

    if (out_idx >= num_outputs) return;

    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int64_t base = outer_idx * dim_size * inner_size + inner_idx;

    // Initialize max_val with the first element
    scalar_t max_val = input[base];

    // Coalesced access: each thread processes consecutive elements
    for (int j = 1; j < dim_size; j++) {
        scalar_t val = input[base + j * inner_size];
        max_val = max(max_val, val);
    }

    // Store the result in shared memory
    shmax[tid] = max_val;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shmax[tid] = max(shmax[tid], shmax[tid + s]);
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (tid == 0) {
        output[blockIdx.x] = shmax[0];
    }
}

// CUDA forward function
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= input.size(i);
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) inner_size *= input.size(i);

    const int64_t dim_size = input.size(dim);
    const int64_t num_outputs = outer_size * inner_size;
    
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    const int threads = 256;
    const int blocks = (num_outputs + threads - 1) / threads;
    size_t shm_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "coalesced_max_reduce_forward", ([&] {
        coalesced_max_reduce_kernel<scalar_t><<<blocks, threads, shm_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            inner_size,
            num_outputs
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Coalesced max reduce forward (CUDA)");
}
