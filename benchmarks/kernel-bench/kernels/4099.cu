#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses vectorized loads (float4) and manually unrolls the computation
// for each of the four elements, reducing loop overhead on the critical path.
// It also unrolls the tail loop using a pragma unroll to minimize branch overhead.

template <typename scalar_t>
__global__ void elu_unroll_kernel(const scalar_t* __restrict__ x,
                                    scalar_t* __restrict__ out,
                                    float alpha,
                                    int n) {
    // Each thread processes 4 elements at a time
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Base index for this thread (processing groups of 4 elements)
    int base = tid * 4 + threadIdx.x;
    // Total stride in terms of elements
    int stride = gridDim.x * blockDim.x * 4;

    // Loop over the input with manual unrolling
    #pragma unroll
    for (int i = base; i < n; i += stride) {
        int remaining = n - i;
        if (remaining >= 4) {
            // Use vectorized load
            float4 vec = *reinterpret_cast<const float4*>(x + i);
            // Manually unroll the ELU computation for each element
            float r0 = (vec.x > 0.f) ? vec.x : alpha * (expf(vec.x) - 1.f);
            float r1 = (vec.y > 0.f) ? vec.y : alpha * (expf(vec.y) - 1.f);
            float r2 = (vec.z > 0.f) ? vec.z : alpha * (expf(vec.z) - 1.f);
            float r3 = (vec.w > 0.f) ? vec.w : alpha * (expf(vec.w) - 1.f);
            
            float4 res = make_float4(r0, r1, r2, r3);
            *reinterpret_cast<float4*>(out + i) = res;
        } else {
            // Handle the tail elements with a manually unrolled loop
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (j < remaining) {
                    float val = x[i + j];
                    out[i + j] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
                }
            }
        }
    }
}

// CUDA wrapper function
torch::Tensor elu_unroll_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    const int n = x.numel();

    // Calculate number of vectorized (float4) operations
    int vec_ops = (n + 3) / 4;
    const int threads = 512;
    const int blocks = (vec_ops + threads - 1) / threads;

    elu_unroll_kernel<float><<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_unroll_cuda, "Vectorized and unrolled ELU activation (CUDA)");
}
