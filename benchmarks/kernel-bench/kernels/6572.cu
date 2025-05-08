#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constant memory for storing dim_size when frequently accessed
__constant__ int64_t const_dim_size;

template <typename scalar_t>
__global__ void max_reduce_constant_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * const_dim_size * inner_size + inner_idx;

        int tid = threadIdx.x;
        int block_size = blockDim.x;
        
        bool valid = false;
        scalar_t thread_max;
        for (int j = tid; j < const_dim_size; j += block_size) {
            scalar_t val = input[base + j * inner_size];
            if (!valid) {
                thread_max = val;
                valid = true;
            } else {
                thread_max = max(thread_max, val);
            }
        }
        
        extern __shared__ char sdata[];
        scalar_t* shmax = reinterpret_cast<scalar_t*>(sdata);
        shmax[tid] = valid ? thread_max : input[base];
        __syncthreads();

        for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shmax[tid] = max(shmax[tid], shmax[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[out_idx] = shmax[0];
        }
    }
}

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
    
    // Use constant memory for the reduction dimension size
    cudaMemcpyToSymbol(const_dim_size, &dim_size, sizeof(int64_t));
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "constant_memory_max_reduce_forward", ([&] {
        int threads = (dim_size < 256 ? dim_size : 256);
        int blocks = (num_outputs < 1024 ? num_outputs : 1024);
        size_t shm_size = threads * sizeof(scalar_t);

        max_reduce_constant_kernel<scalar_t><<<blocks, threads, shm_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            inner_size,
            num_outputs
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA) with constant memory");
}