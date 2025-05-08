#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using stride loops to manage workloads larger than the available threads.
template <typename scalar_t>
__global__ void stride_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t total_elements
) {
    extern __shared__ scalar_t shmax[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize thread's local maximum with a minimal value
    scalar_t local_max = input[0];
    bool valid = false;

    // Stride loop to handle large workloads
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        int outer_idx = i / inner_size;
        int inner_idx = i % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;
        for (int j = 0; j < dim_size; j++) {
            scalar_t val = input[base + j * inner_size];
            if (!valid || val > local_max) {
                local_max = val;
                valid = true;
            }
        }
    }

    // Store local max in shared memory
    shmax[tid] = valid ? local_max : -FLT_MAX;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shmax[tid] = max(shmax[tid], shmax[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = shmax[0];
    }
}

torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    int64_t dim_size = input.size(dim);
    int64_t total_elements = outer_size * inner_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    size_t shm_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "stride_max_reduce_forward", ([&] {
        stride_max_reduce_kernel<scalar_t><<<blocks, threads, shm_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            inner_size,
            total_elements
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Stride max reduce forward (CUDA)");
}
