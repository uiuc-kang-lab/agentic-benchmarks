#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Block size set for efficient occupancy
static const int BLOCK_SIZE = 256;

// This kernel computes the Mean Squared Error (MSE) using vectorized global memory loads
// with __ldg() for read-only accesses, aligning accesses to 128-bit boundaries.
// For float (4 bytes), we use float4 (128-bit) loads, and for double (8 bytes) we use double2 (128-bit) loads.

template <typename scalar_t>
__global__ void vectorized_ldg_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double local_sum = 0.0;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Use vectorized loads if possible
    if (sizeof(scalar_t) == 4) {
        // For float: 4 floats = 16 bytes
        const int vecSize = 4;
        int vecCount = num_elements / vecSize;
        const float4* preds_vec = reinterpret_cast<const float4*>(preds);
        const float4* tgts_vec = reinterpret_cast<const float4*>(tgts);

        // Process vectorized chunks
        for (int i = thread_id; i < vecCount; i += stride) {
            float4 p = __ldg(&preds_vec[i]);
            float4 t = __ldg(&tgts_vec[i]);
            float diff0 = p.x - t.x;
            float diff1 = p.y - t.y;
            float diff2 = p.z - t.z;
            float diff3 = p.w - t.w;
            local_sum += (double)diff0 * diff0 + (double)diff1 * diff1 +
                         (double)diff2 * diff2 + (double)diff3 * diff3;
        }

        // Process any tail elements
        int remainder = num_elements % vecSize;
        int start = vecCount * vecSize;
        for (int i = thread_id; i < remainder; i += stride) {
            float p = __ldg(&preds[start + i]);
            float t = __ldg(&tgts[start + i]);
            float diff = p - t;
            local_sum += (double)diff * diff;
        }
    } else if (sizeof(scalar_t) == 8) {
        // For double: 2 doubles = 16 bytes
        const int vecSize = 2;
        int vecCount = num_elements / vecSize;
        const double2* preds_vec = reinterpret_cast<const double2*>(preds);
        const double2* tgts_vec = reinterpret_cast<const double2*>(tgts);

        // Process vectorized chunks
        for (int i = thread_id; i < vecCount; i += stride) {
            double2 p = __ldg(&preds_vec[i]);
            double2 t = __ldg(&tgts_vec[i]);
            double diff0 = p.x - t.x;
            double diff1 = p.y - t.y;
            local_sum += diff0 * diff0 + diff1 * diff1;
        }

        // Process any tail elements
        int remainder = num_elements % vecSize;
        int start = vecCount * vecSize;
        for (int i = thread_id; i < remainder; i += stride) {
            double p = __ldg(&preds[start + i]);
            double t = __ldg(&tgts[start + i]);
            double diff = p - t;
            local_sum += diff * diff;
        }
    } else {
        // Fallback for other types (shouldn't occur for floating point types)
        for (int i = thread_id; i < num_elements; i += stride) {
            double diff = static_cast<double>(__ldg(&preds[i])) - static_cast<double>(__ldg(&tgts[i]));
            local_sum += diff * diff;
        }
    }

    // Reduce within the block using shared memory
    __shared__ double smem[BLOCK_SIZE];
    smem[threadIdx.x] = local_sum;
    __syncthreads();

    // Standard reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum_out, smem[0]);
    }
}

// Host function that sets up the kernel launch

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size = std::min(grid_size, 1024);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "vectorized_ldg_mse_cuda", ([&] {
        vectorized_ldg_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Final MSE = accumulated squared error divided by number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized LDG MSE forward (CUDA)");
}
