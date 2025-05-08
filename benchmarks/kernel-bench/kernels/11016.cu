#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

// This kernel uses __ldg() to perform read-only vectorized loads aligned to 128-bit boundaries.
// For float data (4 bytes), it loads 4 elements at a time (using float4) and for double (8 bytes),
// it loads 2 elements at a time (using double2). This reduces global memory latency and improves
// throughput while ensuring the correct MSE computation.

template <typename scalar_t>
__global__ void mse_forward_kernel_ldg(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double thread_sum = 0.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // For 32-bit floating point, use vectorized loads with float4 (128-bit load)
    if (sizeof(scalar_t) == 4) {
        const int vec_size = 4; // 4 floats = 16 bytes = 128 bits
        int num_vec = num_elements / vec_size;
        // Process vectorizable portion
        const float4* preds_vec = reinterpret_cast<const float4*>(preds);
        const float4* tgts_vec = reinterpret_cast<const float4*>(tgts);
        for (int i = tid; i < num_vec; i += stride) {
            float4 p = __ldg(&preds_vec[i]);
            float4 t = __ldg(&tgts_vec[i]);
            float diff0 = p.x - t.x;
            float diff1 = p.y - t.y;
            float diff2 = p.z - t.z;
            float diff3 = p.w - t.w;
            thread_sum += (double)(diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3);
        }
        // Process remaining elements
        int start = num_vec * vec_size;
        for (int idx = start + tid; idx < num_elements; idx += stride) {
            float diff = __ldg(&preds[idx]) - __ldg(&tgts[idx]);
            thread_sum += (double)(diff * diff);
        }
    } else if (sizeof(scalar_t) == 8) { // For double precision using double2 (2 doubles = 16 bytes)
        const int vec_size = 2; // 2 doubles = 16 bytes = 128 bits
        int num_vec = num_elements / vec_size;
        const double2* preds_vec = reinterpret_cast<const double2*>(preds);
        const double2* tgts_vec = reinterpret_cast<const double2*>(tgts);
        for (int i = tid; i < num_vec; i += stride) {
            double2 p = __ldg(&preds_vec[i]);
            double2 t = __ldg(&tgts_vec[i]);
            double diff0 = p.x - t.x;
            double diff1 = p.y - t.y;
            thread_sum += diff0 * diff0 + diff1 * diff1;
        }
        // Process remaining elements
        int start = num_vec * vec_size;
        for (int idx = start + tid; idx < num_elements; idx += stride) {
            double diff = __ldg(&preds[idx]) - __ldg(&tgts[idx]);
            thread_sum += diff * diff;
        }
    } else {
        // Fallback for types not handled specifically
        for (int idx = tid; idx < num_elements; idx += stride) {
            double diff = __ldg(&preds[idx]) - __ldg(&tgts[idx]);
            thread_sum += diff * diff;
        }
    }

    // Reduction within block using shared memory
    __shared__ double shm[BLOCK_SIZE];
    int lane = threadIdx.x;
    shm[lane] = thread_sum;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            shm[lane] += shm[lane + offset];
        }
        __syncthreads();
    }

    if (lane == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}


torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));
    
    int device_id;
    cudaGetDevice(&device_id);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    int num_blocks = sm_count * 4; // heuristic: 4 blocks per SM for good occupancy

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_ldg", ([&] {
        mse_forward_kernel_ldg<scalar_t><<<num_blocks, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward using __ldg and vectorized loads (CUDA)");
}
