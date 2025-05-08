#include <torch/extension.h>
#include <vector>
#include <cfloat>

__global__ void block_size_tuning_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    int global_idx = blockIdx.x;
    int outer_idx = global_idx / innerSize;
    int inner_idx = global_idx % innerSize;
    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    extern __shared__ float shared[];
    int* sidx = (int*)&shared[blockDim.x];

    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;

    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        float val = __ldg(&x[base_offset + i * innerSize]);
        if (val > thread_max) {
            thread_max = val;
            thread_max_idx = i;
        }
    }

    shared[threadIdx.x] = thread_max;
    sidx[threadIdx.x] = thread_max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared[threadIdx.x + s] > shared[threadIdx.x]) {
                shared[threadIdx.x] = shared[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) { 
        volatile float* vshared = shared;
        volatile int* vsidx = sidx;
        for (int offset = 32; offset > 0; offset /= 2) {
            if (vshared[threadIdx.x + offset] > vshared[threadIdx.x]) {
                vshared[threadIdx.x] = vshared[threadIdx.x + offset];
                vsidx[threadIdx.x] = vsidx[threadIdx.x + offset];
            }
        }
    }

    if (threadIdx.x == 0) {
        indices[global_idx] = sidx[0];
    }
}

torch::Tensor block_size_tuning_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim.");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) outerSize *= sizes[d];
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) innerSize *= sizes[d];

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) if (d != dim) out_sizes.push_back(sizes[d]);
    auto indices = torch::empty(out_sizes, torch::TensorOptions().device(x.device()).dtype(torch::kLong));

    int blockSizes[] = {32, 64, 128, 256, 512};
    int optimalBlockSize = 128; // Default, but should be determined by experimentation
    for (int bs : blockSizes) {
        // Experiment with different block sizes
        dim3 grid(outerSize * innerSize);
        size_t shared_mem = bs * (sizeof(float) + sizeof(int));

        block_size_tuning_argmax_kernel<<<grid, bs, shared_mem>>>(
            x_contig.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            dimSize,
            innerSize
        );

        // Add synchronization and timing logic here to determine the best block size
        // This is a placeholder for actual benchmarking code
    }

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_size_tuning_argmax_forward_cuda, "Block Size Tuning ArgMax CUDA forward");
}
