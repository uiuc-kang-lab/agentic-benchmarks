#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Stage 1 kernel: Each block computes a partial sum of the Smooth L1 Loss and writes it to a temporary array
__global__ void smooth_l1_loss_stage1(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* partial_sums,
    const int n_elements
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float thread_sum = 0.0f;

    // Process elements in vectorized way using float4 if possible
    int vec_count = n_elements / 4; // number of float4 groups
    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* targ4 = reinterpret_cast<const float4*>(targets);

    for (int i = gid; i < vec_count; i += stride) {
        float4 p = __ldg(pred4 + i);
        float4 t = __ldg(targ4 + i);
        
        float diff = p.x - t.x;
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.y - t.y;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.z - t.z;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        diff = p.w - t.w;
        abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Handle remaining elements
    int scalar_start = vec_count * 4;
    for (int i = scalar_start + gid; i < n_elements; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }

    // Reduction within the block using shared memory
    extern __shared__ float sdata[];
    sdata[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block's partial sum to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Stage 2 kernel: Reduce the partial sums from all blocks to compute the final loss
__global__ void smooth_l1_loss_stage2(
    const float* __restrict__ partial_sums,
    float* output,
    const int num_blocks,
    const int n_elements
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread sums multiple entries if needed
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduce within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Average the loss over all elements
        output[0] = sdata[0] / n_elements;
    }
}

// Host function that launches both kernels

torch::Tensor smooth_l1_loss_two_stage(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(
        predictions.sizes() == targets.sizes(),
        "Input tensors must have the same shape"
    );
    TORCH_CHECK(
        predictions.is_contiguous() && targets.is_contiguous(),
        "Input tensors must be contiguous"
    );
    TORCH_CHECK(
        predictions.device().is_cuda() && targets.device().is_cuda(),
        "Inputs must be CUDA tensors"
    );

    const int n = predictions.numel();

    // Determine block/grid sizes for stage 1 kernel
    const int block_size = 256;
    int grid_size = (n / 4 + block_size - 1) / block_size;
    if (grid_size < 1) grid_size = 1;

    // Allocate temporary tensor for partial sums (one per block)
    auto partial_buffer = torch::empty({grid_size}, predictions.options());

    // Launch stage 1 kernel
    size_t shared_mem_size = block_size * sizeof(float);
    smooth_l1_loss_stage1<<<grid_size, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_buffer.data_ptr<float>(),
        n
    );

    // Launch stage 2 kernel with one block
    int stage2_block_size = (grid_size < 256) ? grid_size : 256;
    shared_mem_size = stage2_block_size * sizeof(float);
    auto output = torch::zeros({1}, predictions.options());
    smooth_l1_loss_stage2<<<1, stage2_block_size, shared_mem_size>>>(
        partial_buffer.data_ptr<float>(),
        output.data_ptr<float>(),
        grid_size,
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_two_stage, "Smooth L1 Loss (CUDA) - Two Stage Reduction Minimizing Atomics");
}
