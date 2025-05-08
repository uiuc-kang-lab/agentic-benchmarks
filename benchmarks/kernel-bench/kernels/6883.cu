#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

__global__ void argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize,
    const int chunk_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;

    if (idx < total) {
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        int start_offset = (outer_idx * dimSize * innerSize) + inner_idx;

        float max_val = x[start_offset];
        int max_idx = 0;

        for (int d = 1; d < dimSize; d++) {
            float val = __ldg(&x[start_offset + d * innerSize]);
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }

        indices[chunk_offset + outer_idx * innerSize + inner_idx] = max_idx;
    }
}

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    auto ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }

    auto options = torch::TensorOptions()
                       .device(x.device())
                       .dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    const int num_streams = 4;  // Number of concurrent streams
    const int chunk_size = (outerSize + num_streams - 1) / num_streams;
    
    cudaStream_t streams[4];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;
    
    for (int i = 0; i < num_streams; i++) {
        int current_chunk_size = std::min(chunk_size, outerSize - i * chunk_size);
        if (current_chunk_size <= 0) break;
        
        const int total = current_chunk_size * innerSize;
        const int blocks = (total + threads - 1) / threads;
        const int chunk_offset = i * chunk_size * innerSize;

        argmax_kernel<<<blocks, threads, 0, streams[i]>>>(
            x_contig.data_ptr<float>() + (i * chunk_size * dimSize * innerSize),
            indices.data_ptr<int64_t>(),
            current_chunk_size,
            dimSize,
            innerSize,
            chunk_offset
        );
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward");
}