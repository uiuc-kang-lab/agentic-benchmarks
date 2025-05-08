#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Store frequently accessed constants in constant memory
__constant__ float kThreshold = 1.0f;
__constant__ float kHalf = 0.5f;

// Kernel using vectorized loads and constant memory for frequently used parameters
__global__ void smooth_l1_loss_const_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Process data in groups of 4 using vectorized loads
    int vec_count = n_elements / 4;
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);

    for (int i = idx; i < vec_count; i += stride) {
        float4 p = __ldg(pred4 + i);
        float4 t = __ldg(targ4 + i);

        float diff = p.x - t.x;
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < kThreshold) ? kHalf * diff * diff : abs_diff - kHalf;

        diff = p.y - t.y;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < kThreshold) ? kHalf * diff * diff : abs_diff - kHalf;

        diff = p.z - t.z;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < kThreshold) ? kHalf * diff * diff : abs_diff - kHalf;

        diff = p.w - t.w;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < kThreshold) ? kHalf * diff * diff : abs_diff - kHalf;
    }

    // Process remaining scalar elements
    int remainder_start = vec_count * 4;
    for (int i = remainder_start + idx; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < kThreshold) ? kHalf * diff * diff : abs_diff - kHalf;
    }

    // Block-level reduction using shared memory
    __shared__ float shared_mem[256];
    shared_mem[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Divide total loss by the number of elements and atomically add to output
        atomicAdd(output, shared_mem[0] / n_elements);
    }
}

// Host function wrapper
torch::Tensor smooth_l1_loss_cuda_const(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n_elements = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    int grid_size = (n_elements / 4 + block_size - 1) / block_size;
    grid_size = grid_size > 0 ? grid_size : 1;

    smooth_l1_loss_const_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda_const, "Smooth L1 Loss (CUDA) with constant memory for parameters");
}
