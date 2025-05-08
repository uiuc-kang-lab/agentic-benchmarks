#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel using warp-level primitives for reduction
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    const unsigned int FULL_MASK = 0xffffffff;
    int tid = threadIdx.x;
    int lane = tid & 31;  // lane id within the warp
    int global_idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    int stride = gridDim.x * blockDim.x;
    
    // Each thread accumulates its partial sum over its strided indices
    for (int i = global_idx; i < numel; i += stride) {
        float val = input[i];
        sum += val * val;
    }
    
    // Perform warp-level reduction using shuffle operations
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    
    // Lane 0 of each warp writes its result to global memory using atomic add
    if (lane == 0 && atomicAdd(norm_out, 0) == 0) {
        atomicAdd(norm_out, sum);
    }
}

// CUDA kernel for normalizing the tensor
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = global_idx; i < numel; i += stride) {
        output[i] = input[i] / norm;
    }
}

// C++ forward function called from Python
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();

    int numel = input.numel();
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    // Compute the sum of squares using warp-level reduction
    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Normalize the tensor
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level reduction based Frobenius norm normalization");
}
