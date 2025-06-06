#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Thread count must be multiple of warp size for optimal performance
constexpr int THREADS_PER_BLOCK = 128;
static_assert(THREADS_PER_BLOCK % 32 == 0, "Thread count must be a multiple of warp size (32)");

// Updated kernel using shared memory
__global__ void gelu_kernel_shared(const float4* __restrict__ __align__(16) x, 
                                 float4* __restrict__ __align__(16) y, 
                                 int vec_size) {
    extern __shared__ float4 shared_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    // Load data into shared memory
    if (i < vec_size) {
        shared_data[tid] = x[i];
    }
    __syncthreads(); // Ensure all loads to shared memory are done

    if (i < vec_size) {
        float4 v = shared_data[tid];
        
        // Compute GELU on each component
        float x0 = v.x;
        float x1 = v.y;
        float x2 = v.z;
        float x3 = v.w;

        float x0_cubed = x0 * x0 * x0;
        float inner0 = (x0 + coeff * x0_cubed) * sqrt_2_over_pi;
        float y0 = 0.5f * x0 * (1.0f + tanhf(inner0));

        float x1_cubed = x1 * x1 * x1;
        float inner1 = (x1 + coeff * x1_cubed) * sqrt_2_over_pi;
        float y1 = 0.5f * x1 * (1.0f + tanhf(inner1));

        float x2_cubed = x2 * x2 * x2;
        float inner2 = (x2 + coeff * x2_cubed) * sqrt_2_over_pi;
        float y2 = 0.5f * x2 * (1.0f + tanhf(inner2));

        float x3_cubed = x3 * x3 * x3;
        float inner3 = (x3 + coeff * x3_cubed) * sqrt_2_over_pi;
        float y3 = 0.5f * x3 * (1.0f + tanhf(inner3));
        
        float4 out;
        out.x = y0;
        out.y = y1;
        out.z = y2;
        out.w = y3;
        y[i] = out;
    }
}

// Fallback scalar kernel for remaining elements
__global__ void gelu_kernel_scalar(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        float xi = __ldg(&x[i]);
        float x_cubed = xi * xi * xi;
        float inner = (xi + coeff * x_cubed) * sqrt_2_over_pi;
        y[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    // Process most of the tensor with vectorized float4 loads/stores
    int vec_size = n / 4;  // number of float4 vectors
    int remainder = n % 4;

    const int threads = 128;
    if(vec_size > 0) {
        int blocks = (vec_size + threads - 1) / threads;
        const float4* x_vec = reinterpret_cast<const float4*>(x.data_ptr<float>());
        float4* y_vec = reinterpret_cast<float4*>(y.data_ptr<float>());
        gelu_kernel_shared<<<blocks, threads, threads * sizeof(float4)>>>(x_vec, y_vec, vec_size);
    }

    // Process any remaining elements with the scalar kernel
    if(remainder > 0) {
        int offset = vec_size * 4;
        int blocks_rem = (remainder + threads - 1) / threads;
        gelu_kernel_scalar<<<blocks_rem, threads>>>(x.data_ptr<float>() + offset, y.data_ptr<float>() + offset, remainder);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU shared memory forward CUDA implementation");
}