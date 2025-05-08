#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>

// Helper function: warp-level reduction of a float value
__inline__ __device__ float warpReduceSum(float val) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
    for (int offset = warpSize/2; offset > 0; offset /= 2)
         val += __shfl_down_sync(0xffffffff, val, offset);
#else
    for (int offset = warpSize/2; offset > 0; offset /= 2)
         val += __shfl_down(val, offset);
#endif
    return val;
}

// Helper function: block-level reduction using warpReduceSum
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum(val);
    __syncthreads();
    if(lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if(wid == 0) val = warpReduceSum(val);
    return val;
}

// Kernel that leverages shared memory for instance normalization when appropriate
// Each block processes one instance (i.e. one (n, c) pair) over H*W elements
// If HW is small enough (<= SHARED_MEM_THRESHOLD) the kernel loads the instance into shared memory,
// performs a parallel reduction to compute mean and variance, then normalizes the values using the staged data.
// Otherwise, it falls back to a vectorized global memory approach.

__global__ void instance_norm_kernel_shared(
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
    const int instance_id = blockIdx.x;
    if (instance_id >= N * C) return;

    const int n = instance_id / C;
    const int c = instance_id % C;
    const int HW = H * W;
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    // Threshold for using shared memory branch
    const int SHARED_MEM_THRESHOLD = 4096;

    if (HW <= SHARED_MEM_THRESHOLD) {
        // Use shared memory to stage the input instance
        // Dynamic shared memory layout: first HW floats for instance data,
        // then blockDim.x floats for partial sums and blockDim.x for partial squared sums
        extern __shared__ float s[];
        float* s_data = s;                     // size: HW
        float* s_partial_sum = s + HW;         // size: blockDim.x
        float* s_partial_sum_sq = s + HW + blockDim.x; // size: blockDim.x

        // Load the entire instance from global memory into shared memory
        for (int i = threadIdx.x; i < HW; i += blockDim.x) { if (i < HW) {
            s_data[i] = x_ptr[i];
        }
        __syncthreads();

        // Each thread computes a local partial sum and partial sum of squares from shared memory
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HW; i += blockDim.x) { if (i < HW) {
            float val = s_data[i];
            local_sum += val;
            local_sum_sq += val * val;
        }
        s_partial_sum[threadIdx.x] = local_sum;
        s_partial_sum_sq[threadIdx.x] = local_sum_sq;
        __syncthreads();

        // Parallel reduction over the partial sums in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                s_partial_sum[threadIdx.x] += s_partial_sum[threadIdx.x + stride];
                s_partial_sum_sq[threadIdx.x] += s_partial_sum_sq[threadIdx.x + stride];
            }
            __syncthreads();
        }

        float mean = s_partial_sum[0] / HW;
        float var = s_partial_sum_sq[0] / HW - mean * mean;
        var = fmaxf(var, 0.0f);
        float inv_std = rsqrtf(var + eps);
        
        const float w = (weight != nullptr) ? __ldg(weight + c) : 1.0f;
        const float b = (bias != nullptr) ? __ldg(bias + c) : 0.0f;
        __syncthreads();

        // Normalize the staged data in shared memory
        for (int i = threadIdx.x; i < HW; i += blockDim.x) { if (i < HW) {
            float val = s_data[i];
            s_data[i] = (val - mean) * inv_std * w + b;
        }
        __syncthreads();

        // Write the normalized values from shared memory back to global memory
        for (int i = threadIdx.x; i < HW; i += blockDim.x) { if (i < HW) {
            y_ptr[i] = s_data[i];
        }

    } else {
        // Fallback branch: use existing vectorized global memory approach if instance is large
        float sum_val = 0.0f, sum_sq_val = 0.0f;
        const int vector_size = 4;
        int items_per_thread = (HW + (blockDim.x * vector_size) - 1) / (blockDim.x * vector_size);
        int aligned_HW = (HW / vector_size) * vector_size;
        
        for (int item = 0; item < items_per_thread; ++item) {
            int idx = (item * blockDim.x + threadIdx.x) * vector_size;
            if (idx < aligned_HW) {
                float4 vec = __ldg(reinterpret_cast<const float4*>(x_ptr + idx));
                sum_val += vec.x + vec.y + vec.z + vec.w;
                sum_sq_val += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
            }
        }
        for (int i = aligned_HW + threadIdx.x; i < HW; i += blockDim.x) {
            float v = __ldg(x_ptr + i);
            sum_val += v;
            sum_sq_val += v * v;
        }
        
        sum_val = blockReduceSum(sum_val);
        sum_sq_val = blockReduceSum(sum_sq_val);
        
        __shared__ float sharedMean, sharedInvStd;
        if (threadIdx.x == 0) {
            float mean = sum_val / HW;
            float var = fmaxf(sum_sq_val / HW - mean * mean, 0.0f);
            sharedMean = mean;
            sharedInvStd = rsqrtf(var + eps);
        }
        __syncthreads();
        float mean = sharedMean;
        float inv_std = sharedInvStd;
        const float w = (weight != nullptr) ? __ldg(weight + c) : 1.0f;
        const float b = (bias != nullptr) ? __ldg(bias + c) : 0.0f;
        
        for (int i = threadIdx.x; i < HW; i += blockDim.x) { if (i < HW) {
            float v = __ldg(x_ptr + i);
            y_ptr[i] = (v - mean) * inv_std * w + b;
        }
    }
}

// Host function forwarded to Python via Pybind11
// It selects the shared memory optimized branch for small instances and falls back otherwise.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input must be 4D: (N, C, H, W)");
    
    const int N = sizes[0];
    const int C = sizes[1];
    const int H = sizes[2];
    const int W = sizes[3];
    const int HW = H * W;
    auto y = torch::empty_like(x);
    const int blocks = N * C;
    const int threads = 256;
    
    // Use the shared memory branch if HW is small enough.
    const int SHARED_MEM_THRESHOLD = 4096; // threshold in elements
    if (HW <= SHARED_MEM_THRESHOLD) {
        // Allocate dynamic shared memory: space for instance data + 2 * threads for partial sums
        size_t shared_bytes = (HW + 2 * threads) * sizeof(float);
        instance_norm_kernel_shared<<<blocks, threads, shared_bytes>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
            (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
            N, C, H, W, static_cast<float>(eps)
        );
    } else {
        instance_norm_kernel_shared<<<blocks, threads, 0>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
            (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
            N, C, H, W, static_cast<float>(eps)
        );
    }
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "InstanceNorm shared memory optimized (CUDA)");
}
