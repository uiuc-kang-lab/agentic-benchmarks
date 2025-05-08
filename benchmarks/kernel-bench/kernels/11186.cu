#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Aligned structure for vectorized loads
struct __align__(16) AlignedFloat4 {
    float x, y, z, w;
};

__global__ void smooth_l1_loss_aligned_kernel(
    const AlignedFloat4* __restrict__ predictions4,
    const AlignedFloat4* __restrict__ targets4,
    float* output,
    const int n_vec_elements,
    const float* __restrict__ pred_remainder,
    const float* __restrict__ targ_remainder,
    const int remainder_elements
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float thread_sum = 0.0f;

    // Process aligned float4 elements
    for (int i = gid; i < n_vec_elements; i += gridDim.x * blockDim.x) {
        // Use __ldg for read-only data with 128-bit aligned access
        const AlignedFloat4 pred = __ldg(&predictions4[i]);
        const AlignedFloat4 targ = __ldg(&targets4[i]);

        // Process x component
        float diff = pred.x - targ.x;
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        // Process y component
        diff = pred.y - targ.y;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        // Process z component
        diff = pred.z - targ.z;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        // Process w component
        diff = pred.w - targ.w;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Process remaining elements
    for (int i = gid; i < remainder_elements; i += gridDim.x * blockDim.x) {
        float diff = __ldg(&pred_remainder[i]) - __ldg(&targ_remainder[i]);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Efficient block reduction using aligned shared memory
    __shared__ __align__(16) float shared_mem[256];
    shared_mem[tid] = thread_sum;
    __syncthreads();

    // Reduce within warps first
    if (tid < 128) shared_mem[tid] += shared_mem[tid + 128];
    __syncthreads();
    if (tid < 64) shared_mem[tid] += shared_mem[tid + 64];
    __syncthreads();
    
    // Last warp reduces
    if (tid < 32) {
        volatile float* smem = shared_mem;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    // Add block result to global output
    if (tid == 0) {
        atomicAdd(output, shared_mem[0] / (n_vec_elements * 4 + remainder_elements));
    }
}

torch::Tensor smooth_l1_loss_aligned(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    const int n_elements = predictions.numel();
    const int n_vec_elements = n_elements / 4;
    const int remainder_elements = n_elements % 4;
    auto output = torch::zeros({1}, predictions.options());

    // Calculate optimal grid size
    const int block_size = 256;
    const int grid_size = std::min(65535, (n_vec_elements + block_size - 1) / block_size);

    // Get data pointers
    const float* pred_ptr = predictions.data_ptr<float>();
    const float* targ_ptr = targets.data_ptr<float>();

    // Cast to aligned float4 pointers
    const AlignedFloat4* pred4 = reinterpret_cast<const AlignedFloat4*>(pred_ptr);
    const AlignedFloat4* targ4 = reinterpret_cast<const AlignedFloat4*>(targ_ptr);

    // Get pointers for remainder
    const float* pred_remainder = pred_ptr + (n_vec_elements * 4);
    const float* targ_remainder = targ_ptr + (n_vec_elements * 4);

    smooth_l1_loss_aligned_kernel<<<grid_size, block_size>>>(
        pred4,
        targ4,
        output.data_ptr<float>(),
        n_vec_elements,
        pred_remainder,
        targ_remainder,
        remainder_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_aligned, "Aligned Smooth L1 Loss (CUDA)");
}