#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void adaptive_min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

    int idx = blockIdx.x;
    if (idx >= outer * inner) return;

    int outer_idx = idx / inner;
    int inner_idx = idx % inner;
    const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;

    // Each thread computes partial minimum
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    
    #pragma unroll 4
    for (int j = threadIdx.x; j < r; j += BLOCK_SIZE) {
        scalar_t val = in_ptr[j * inner];
        local_min = (val < local_min) ? val : local_min;
    }

    // Shared memory allocation
    __shared__ scalar_t sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_min;
    __syncthreads();

    // Reduction in shared memory
    if (BLOCK_SIZE >= 512) { if (threadIdx.x < 256) { if (sdata[threadIdx.x + 256] < sdata[threadIdx.x]) sdata[threadIdx.x] = sdata[threadIdx.x + 256]; } __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (threadIdx.x < 128) { if (sdata[threadIdx.x + 128] < sdata[threadIdx.x]) sdata[threadIdx.x] = sdata[threadIdx.x + 128]; } __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (threadIdx.x < 64) { if (sdata[threadIdx.x + 64] < sdata[threadIdx.x]) sdata[threadIdx.x] = sdata[threadIdx.x + 64]; } __syncthreads(); }
    
    if (threadIdx.x < 32) {
        // Warp-level reduction using shuffle
        volatile scalar_t* vsdata = sdata;
        if (BLOCK_SIZE >= 64) vsdata[threadIdx.x] = (vsdata[threadIdx.x + 32] < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 32] : vsdata[threadIdx.x];
        vsdata[threadIdx.x] = (vsdata[threadIdx.x + 16] < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 16] : vsdata[threadIdx.x];
        vsdata[threadIdx.x] = (vsdata[threadIdx.x + 8] < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 8] : vsdata[threadIdx.x];
        vsdata[threadIdx.x] = (vsdata[threadIdx.x + 4] < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 4] : vsdata[threadIdx.x];
        vsdata[threadIdx.x] = (vsdata[threadIdx.x + 2] < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 2] : vsdata[threadIdx.x];
        vsdata[threadIdx.x] = (vsdata[threadIdx.x + 1] < vsdata[threadIdx.x]) ? vsdata[threadIdx.x + 1] : vsdata[threadIdx.x];
    }

    if (threadIdx.x == 0) {
        output[idx] = sdata[0];
    }
}

torch::Tensor forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int ndim = input.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    int outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= input.size(i);
    }
    int r = input.size(dim);
    int inner = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner *= input.size(i);
    }

    std::vector<int64_t> output_shape;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            output_shape.push_back(input.size(i));
        }
    }

    auto output = torch::empty(output_shape, input.options());
    int total = outer * inner;

    // Adaptive block size selection based on reduction dimension size
    int block_size;
    if (r <= 64) block_size = 32;
    else if (r <= 128) block_size = 64;
    else if (r <= 256) block_size = 128;
    else block_size = 256;

    int blocks = total;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "adaptive_min_reduce_cuda", ([&] {
        switch(block_size) {
            case 32:
                adaptive_min_reduction_kernel<scalar_t, 32><<<blocks, 32, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), outer, r, inner);
                break;
            case 64:
                adaptive_min_reduction_kernel<scalar_t, 64><<<blocks, 64, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), outer, r, inner);
                break;
            case 128:
                adaptive_min_reduction_kernel<scalar_t, 128><<<blocks, 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), outer, r, inner);
                break;
            default:
                adaptive_min_reduction_kernel<scalar_t, 256><<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), outer, r, inner);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive block size min reduction (CUDA)");
}