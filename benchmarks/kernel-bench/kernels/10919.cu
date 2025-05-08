#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size
static const int BLOCK_SIZE = 256;

// CUDA kernel for computing MSE loss using __ldg() and 128-bit aligned vectorized loads

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double thread_sum = 0.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Vectorized load for 32-bit types (e.g., float): load 4 elements (128-bit) at a time
    if constexpr (sizeof(scalar_t) == 4) {
        // Determine number of elements which can be processed in vectorized manner
        int num_vectorized = (num_elements / 4) * 4;
        const float4* preds4 = reinterpret_cast<const float4*>(preds);
        const float4* tgts4 = reinterpret_cast<const float4*>(tgts);
        int vector_length = num_vectorized / 4;

        for (int i = tid; i < vector_length; i += stride) {
            // Use __ldg to load from read-only cache
            float4 p = __ldg(&preds4[i]);
            float4 t = __ldg(&tgts4[i]);
            float diff0 = p.x - t.x;
            float diff1 = p.y - t.y;
            float diff2 = p.z - t.z;
            float diff3 = p.w - t.w;
            thread_sum += (double)(diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3);
        }
        // Process any remaining elements
        for (int i = num_vectorized + tid; i < num_elements; i += stride) {
            float p_val = __ldg(preds + i);
            float t_val = __ldg(tgts + i);
            float diff = p_val - t_val;
            thread_sum += (double)(diff * diff);
        }
    }
    // Vectorized load for 64-bit types (e.g., double): load 2 elements (128-bit) at a time
    else if constexpr (sizeof(scalar_t) == 8) {
        int num_vectorized = (num_elements / 2) * 2;
        const double2* preds2 = reinterpret_cast<const double2*>(preds);
        const double2* tgts2 = reinterpret_cast<const double2*>(tgts);
        int vector_length = num_vectorized / 2;

        for (int i = tid; i < vector_length; i += stride) {
            double2 p = __ldg(&preds2[i]);
            double2 t = __ldg(&tgts2[i]);
            double diff0 = p.x - t.x;
            double diff1 = p.y - t.y;
            thread_sum += diff0 * diff0 + diff1 * diff1;
        }
        // Process tail elements
        for (int i = num_vectorized + tid; i < num_elements; i += stride) {
            double p_val = __ldg(preds + i);
            double t_val = __ldg(tgts + i);
            double diff = p_val - t_val;
            thread_sum += diff * diff;
        }
    } else {
        // Fallback for other scalar types (shouldn't occur for typical MSE loss inputs)
        for (int i = tid; i < num_elements; i += stride) {
            scalar_t p_val = __ldg(preds + i);
            scalar_t t_val = __ldg(tgts + i);
            double diff = static_cast<double>(p_val) - static_cast<double>(t_val);
            thread_sum += diff * diff;
        }
    }

    // Reduction within block using shared memory
    __shared__ double shared_data[BLOCK_SIZE];
    int local_tid = threadIdx.x;
    shared_data[local_tid] = thread_sum;
    __syncthreads();

    // Perform iterative reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (local_tid < s) {
            shared_data[local_tid] += shared_data[local_tid + s];
        }
        __syncthreads();
    }

    // Atomic add the block sum to the global accumulator
    if (local_tid == 0) {
        atomicAdd(sum_out, shared_data[0]);
    }
}

// Host function wrapping the CUDA kernel call

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Launch configuration using BLOCK_SIZE threads per block
    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", ([&] {
        mse_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Final mean squared error computation
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA)");
}
