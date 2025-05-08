#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Warp-level reduction using shuffle
__inline__ __device__ float warpReduceSum(float val) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
#else
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
#endif
    return val;
}

// Block-level reduction using a statically allocated shared memory buffer
__inline__ __device__ float blockReduceSum(float val) {
    // Note: This static shared memory is allocated in addition to dynamic shared memory used in the kernel
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    __syncthreads();

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// CUDA kernel for Instance Normalization with dynamic block size tuning
__global__ void instance_norm_kernel_opt(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N,
    int C,
    int H,
    int W,
    float eps
) {
    // Allocate dynamic shared memory for temporary storage of input values
    extern __shared__ float shared_data[];
    float* temp_storage = shared_data;

    int instance_id = blockIdx.x; // One block per instance (n, c)
    int n = instance_id / C;
    int c = instance_id % C;
    int HW = H * W;

    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Each thread loads elements, storing into shared memory and accumulating sum and sum of squares
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float val = x_ptr[i];
        temp_storage[i] = val;
        sum += val;
        sum_sq += val * val;
    }

    // Reduce within block to compute total sum
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float mean_sh;
    __shared__ float invstd_sh;
    if (threadIdx.x == 0) {
        float mean = sum / HW;
        float var = sum_sq / HW - mean * mean;
        var = (var < 0.f) ? 0.f : var;
        mean_sh = mean;
        invstd_sh = rsqrtf(var + eps);
    }
    __syncthreads();

    // Load scale and bias if provided, otherwise use defaults
    float scale = (weight != nullptr) ? weight[c] : 1.0f;
    float shift = (bias != nullptr) ? bias[c] : 0.0f;

    // Normalize and apply scale/shift
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float val = temp_storage[i];
        val = (val - mean_sh) * invstd_sh;
        y_ptr[i] = val * scale + shift;
    }
}

// Forward function called from Python
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    if (weight.defined()) {
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    }
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    }

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");
    
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    auto y = torch::empty_like(x);

    int HW = H * W;
    int block_size;
    // Simple heuristic to choose optimal block size based on the size of each instance
    if (HW <= 256) {
        block_size = 128;
    } else if (HW <= 1024) {
        block_size = 64;
    } else if (HW <= 4096) {
        block_size = 128;
    } else if (HW <= 16384) {
        block_size = 256;
    } else {
        block_size = 512;
    }

    int blocks = N * C;  // One block per (n, c) instance
    int shared_mem_size = HW * sizeof(float);

    instance_norm_kernel_opt<<<blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );

    return y;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA) with block size tuning");
}
