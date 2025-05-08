/*
 * This CUDA extension computes the frobenius norm normalization using an improved reduction scheme.
 * It leverages vectorized memory accesses (float4) and warp-level reduction using __shfl_down_sync for
 * efficient intra-warp summation, reducing shared memory usage and synchronization overhead.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    // Assuming full 32-bit warp mask
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel to compute the sum of squares (squared Frobenius norm) with vectorized loads and warp reduction
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    float sum = 0.0f;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Process main part using 128-bit (float4) loads if possible
    int n_vec = numel / 4;  // number of float4 elements
    for (int i = global_tid; i < n_vec; i += stride) {
        // Use __ldg for read-only cache
        float4 v = __ldg(reinterpret_cast<const float4*>(input) + i);
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Process any remaining elements
    int rem_start = n_vec * 4;
    for (int i = rem_start + global_tid; i < numel; i += stride) {
        float val = __ldg(input + i);
        sum += val * val;
    }

    // Intra-warp reduction
    sum = warpReduceSum(sum);

    // Use shared memory to reduce one value per warp
    __shared__ float shared[32];  // maximum number of warps per block (256/32 = 8 typically)
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();

    // Let the first warp reduce the sums from each warp
    float blockSum = 0.0f;
    if (threadIdx.x < 32) {  // Only first warp participates
        // Load value if this thread's lane corresponds to a valid warp
        blockSum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        blockSum = warpReduceSum(blockSum);
        
        // Only thread 0 needs to do the atomic add
        if (lane == 0) {
            atomicAdd(norm_out, blockSum);
        }
    }
}

// Kernel to normalize the input tensor using vectorized loads/stores
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n_vec = numel / 4;  // number of float4 elements

    // Process main vectorized part
    for (int i = global_tid; i < n_vec; i += stride) {
        float4 in_val = __ldg(reinterpret_cast<const float4*>(input) + i);
        float4 out_val;
        out_val.x = in_val.x / norm;
        out_val.y = in_val.y / norm;
        out_val.z = in_val.z / norm;
        out_val.w = in_val.w / norm;
        reinterpret_cast<float4*>(output)[i] = out_val;
    }

    // Process remaining elements
    int rem_start = n_vec * 4;
    for (int i = rem_start + global_tid; i < numel; i += stride) {
        output[i] = __ldg(input + i) / norm;
    }
}

// Forward function called from Python
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

    // Compute squared norm using vectorized loads and efficient warp reduction
    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    // Copy norm from device to host and compute the square root
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Normalize the tensor using optimized vectorized operations
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient Frobenius norm normalization using vectorized loads and warp-level reduction");
}
