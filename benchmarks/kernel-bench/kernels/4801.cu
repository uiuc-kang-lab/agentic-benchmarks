#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 32

template <unsigned int WF>
__device__ float warpReduceSum(float val) {
    for (int offset = WF/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float computeStridedSum(const float* input, int numel) {
    float sum = 0.0f;
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Process 4 elements at once using vectorized loads
    const float4* vec_input = reinterpret_cast<const float4*>(input);
    const int vec_size = 4;
    const int n_full_vec = numel / vec_size;
    
    for (int i = global_idx; i < n_full_vec; i += stride) {
        float4 v = vec_input[i];
        sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    // Handle remaining elements
    const int base_idx = n_full_vec * vec_size;
    for (int i = base_idx + global_idx; i < numel; i += stride) {
        sum += input[i] * input[i];
    }
    return sum;
}

__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    const float sum = computeStridedSum(input, numel);
    const float reduced = warpReduceSum<32>(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(norm_out, reduced);
    }
}

__constant__ float d_norm;

__global__ void normalize_kernel(const float* input, float* output, int numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / max(1e-8f, d_norm);
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    const int numel = input.numel();
    const int blocks = min(4096, (numel + BLOCK_SIZE*32 - 1) / (BLOCK_SIZE*32));

    compute_norm_kernel<<<blocks, BLOCK_SIZE>>>(input.data_ptr<float>(),
                                              norm_tensor.data_ptr<float>(),
                                              numel);

    float norm_val;
    cudaMemcpy(&norm_val, norm_tensor.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val);

    cudaMemcpyToSymbol(d_norm, &norm_val, sizeof(float), 0, cudaMemcpyHostToDevice);

    const int norm_blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    normalize_kernel<<<norm_blocks, BLOCK_SIZE>>>(input.data_ptr<float>(),
                                                output.data_ptr<float>(),
                                                numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Frobenius norm with warp reductions");
}