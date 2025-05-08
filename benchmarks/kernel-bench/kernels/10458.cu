#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized cumulative sum kernel using both grid-stride and thread-stride loops
__global__ void cumsum_kernel_opt(const float* __restrict__ input, float* output,
                                    int outer_size, int inner_size, int stride) {
    // Use grid-stride loop in case outer_size exceeds gridDim.x
    for (int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        // Each thread processes several inner indices if needed
        for (int inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
            float sum = 0.0f;
            int base = outer_idx * stride * inner_size + inner_idx;
            
            // Unroll cumulative sum loop along the 'stride' dimension
            #pragma unroll 16
            for (int s = 0; s < stride; ++s) {
                int idx = base + s * inner_size;
                sum += __ldg(&input[idx]);  // Utilize read-only cache
                output[idx] = sum;
            }
        }
    }
}

// Host function to set up dimensions and launch the kernel
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= x.size(i);
    }

    int stride = x.size(dim);

    // Set thread count as min(inner_size, 256) for efficiency
    int threads = (inner_size < 256) ? inner_size : 256;
    // Use grid-stride loop for outer dimension; cap blocks to 1024 if outer_size is huge
    int blocks = (outer_size < 1024) ? outer_size : 1024;

    cumsum_kernel_opt<<<blocks, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), outer_size, inner_size, stride
    );

    // Optional: synchronize for correctness checking (can be removed in production)
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA cumulative sum with grid and thread striding");
}
