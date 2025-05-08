#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Warp-level reduction using shuffle intrinsics
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

// Block-level reduction that combines warp reduction and shared memory
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // one element per warp (max 32 warps per block)
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    // Each warp reduces its own subset
    val = warpReduceSum(val);
    if (lane == 0) {
        shared[warpId] = val;
    }
    __syncthreads();

    // Only the first warp loads the partial sums
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
    if (warpId == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Optimized Instance Normalization kernel that fuses dynamic block size tuning with efficient reduction
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
    // Allocate dynamic shared memory for caching pixel values
    extern __shared__ float temp_storage[];

    // Each block handles one (n, c) instance
    int instance_id = blockIdx.x;
    int n = instance_id / C;
    int c = instance_id % C;
    int HW = H * W;

    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Use cooperative groups for better load balancing and async operations
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    
    // Pipeline the loads using registers to overlap memory operations
    float val = 0.0f;
    float next_val = 0.0f;
    if (threadIdx.x < HW) {
        next_val = x_ptr[threadIdx.x];
    }
    
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        val = next_val;
        if (i + blockDim.x < HW) {
            next_val = x_ptr[i + blockDim.x];
        }
        temp_storage[i] = val;
        sum += val;
        sum_sq += val * val;
        
        // Use cooperative groups synchronization for better performance
        block.sync();

    // Compute the total sum and sum of squares for the instance
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float mean;
    __shared__ float invstd;
    if (threadIdx.x == 0) {
        mean = sum / HW;
        float var = (sum_sq / HW) - (mean * mean);
        var = (var < 0.f) ? 0.f : var;  // guard against negative variance
        invstd = rsqrtf(var + eps);
    }
    __syncthreads();

    // Load scale and bias per channel if provided
    float scale = (weight != nullptr) ? weight[c] : 1.0f;
    float shift = (bias != nullptr) ? bias[c] : 0.0f;

    // Normalize the inputs using the computed mean and variance
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float val = temp_storage[i];
        y_ptr[i] = ((val - mean) * invstd) * scale + shift;
    }
}

// Forward function interfacing with PyTorch
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    if (weight.defined() && weight.numel() > 0)
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined() && bias.numel() > 0)
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    auto y = torch::empty_like(x);

    // Dynamically choose block size based on the spatial dimensions
    int HW = H * W;
    int block_size;
    if (HW <= 256) {
        block_size = 32;
    } else if (HW <= 1024) {
        block_size = 64;
    } else if (HW <= 4096) {
        block_size = 128;
    } else if (HW <= 16384) {
        block_size = 256;
    } else {
        block_size = 512;
    }

    int blocks = N * C;  // one block per instance (n, c)
    int shared_mem_size = HW * sizeof(float);  // allocate shared memory for input values

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
    m.def("forward", &forward, "Optimized Instance Normalization forward (CUDA)");
}
