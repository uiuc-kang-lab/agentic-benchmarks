#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__constant__ float d_alpha;

template<typename scalar_t>
__global__ void elu_const_kernel(const scalar_t* __restrict__ x,
                                scalar_t* __restrict__ out,
                                int n) {
    extern __shared__ float s_data[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    // Number of elements per thread
    const int ELEMENTS_PER_THREAD = 4;
    // Shared memory tile size (elements per block)
    const int TILE_SIZE = blockDim.x * ELEMENTS_PER_THREAD;
    
    for (int base = bid * TILE_SIZE; base < n; base += stride * ELEMENTS_PER_THREAD) {
        // Load data into shared memory
        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            int idx = base + tid * ELEMENTS_PER_THREAD + j;
            if (idx < n) {
                s_data[tid * ELEMENTS_PER_THREAD + j] = x[idx];
            }
        }
        __syncthreads();
        
        // Process data in shared memory
        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            int idx = base + tid * ELEMENTS_PER_THREAD + j;
            if (idx < n) {
                float val = s_data[tid * ELEMENTS_PER_THREAD + j];
                s_data[tid * ELEMENTS_PER_THREAD + j] = (val > 0) ? val : d_alpha * (expf(val) - 1);
            }
        }
        __syncthreads();
        
        // Write results back to global memory
        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            int idx = base + tid * ELEMENTS_PER_THREAD + j;
            if (idx < n) {
                out[idx] = s_data[tid * ELEMENTS_PER_THREAD + j];
            }
        }
        __syncthreads();
    }
}

torch::Tensor elu_const_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    const int n = x.numel();

    // Copy alpha to constant memory
    cudaMemcpyToSymbol(d_alpha, &alpha, sizeof(float));

    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);

    elu_const_kernel<float><<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_const_cuda, "Constant memory ELU activation (CUDA)");
}