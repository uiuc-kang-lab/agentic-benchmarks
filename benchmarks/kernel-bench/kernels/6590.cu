#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store frequently accessed, read-only parameters (dim_size and inner_size) in constant memory
// d_const_params[0] = dim_size, d_const_params[1] = inner_size
__constant__ int64_t d_const_params[2];

// Templated kernel using a compile-time BLOCK_SIZE that reads dimension info from constant memory
template <typename scalar_t, int BLOCK_SIZE>
__global__ void const_mem_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t num_outputs
) {
    // Load constant parameters
    const int64_t dim_size = d_const_params[0];
    const int64_t inner_size = d_const_params[1];

    // Process each output element using a grid-stride loop
    for (int out_idx = blockIdx.x; out_idx < num_outputs; out_idx += gridDim.x) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        int tid = threadIdx.x;
        int block_size = blockDim.x;
        bool valid = false;
        scalar_t thread_max;

        // Each thread processes several elements in the reduction dimension
        for (int j = tid; j < dim_size; j += block_size) {
            scalar_t val = input[base + j * inner_size];
            if (!valid) {
                thread_max = val;
                valid = true;
            } else {
                thread_max = max(thread_max, val);
            }
        }

        // Reduction in shared memory
        extern __shared__ char sdata[];
        scalar_t* shmem = reinterpret_cast<scalar_t*>(sdata);
        shmem[tid] = valid ? thread_max : input[base];
        __syncthreads();

        // Tree-based reduction
        for (int stride = block_size / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shmem[tid] = max(shmem[tid], shmem[tid + stride]);
            }
            __syncthreads();
        }

        // Write the result for this output element
        if (tid == 0) {
            output[out_idx] = shmem[0];
        }
    }
}

// Forward function that sets constant memory and launches the kernel
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Adjust negative dimension
    if (dim < 0) dim += input.dim();

    // Calculate outer_size (product of dimensions before 'dim')
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    // Calculate inner_size (product of dimensions after 'dim')
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    // Get the size for the reduction dimension
    const int64_t dim_size = input.size(dim);
    const int64_t num_outputs = outer_size * inner_size;

    // Prepare the output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Copy the frequently used parameters into constant memory
    int64_t const_params[2];
    const_params[0] = dim_size;
    const_params[1] = inner_size;
    cudaMemcpyToSymbol(d_const_params, const_params, sizeof(const_params), 0, cudaMemcpyHostToDevice);

    // Choose block size heuristically based on dim_size
    int block_size = 256;
    if (dim_size <= 32) {
        block_size = 32;
    } else if (dim_size <= 64) {
        block_size = 64;
    } else if (dim_size <= 128) {
        block_size = 128;
    } else if (dim_size <= 256) {
        block_size = 256;
    } else {
        block_size = 512;
    }

    if (num_outputs < block_size) {
        block_size = num_outputs;
    }

    // Limit the grid size to balance kernel launch overhead
    int blocks = (num_outputs < 1024) ? num_outputs : 1024;

    // Calculate shared memory size (per block)
    size_t shared_mem_size = block_size * input.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "const_mem_max_reduce_forward", ([&] {
        switch (block_size) {
            case 32:
                const_mem_max_reduce_kernel<scalar_t, 32><<<blocks, block_size, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
            case 64:
                const_mem_max_reduce_kernel<scalar_t, 64><<<blocks, block_size, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
            case 128:
                const_mem_max_reduce_kernel<scalar_t, 128><<<blocks, block_size, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
            case 256:
                const_mem_max_reduce_kernel<scalar_t, 256><<<blocks, block_size, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
            case 512:
                const_mem_max_reduce_kernel<scalar_t, 512><<<blocks, block_size, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
            default:
                const_mem_max_reduce_kernel<scalar_t, 256><<<blocks, block_size, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    num_outputs);
                break;
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA) using constant memory for parameters");
}
