#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel remains optimized for vectorized processing
__global__ void pipelined_smooth_l1_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int chunk_size,
    int chunk_offset
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;

    // Process chunk using vectorized loads
    int vec_count = chunk_size / 4;
    const float4* pred4 = reinterpret_cast<const float4*>(predictions + chunk_offset);
    const float4* targ4 = reinterpret_cast<const float4*>(targets + chunk_offset);

    for (int i = idx; i < vec_count; i += stride) {
        float4 p = __ldg(pred4 + i);
        float4 t = __ldg(targ4 + i);

        float diff = p.x - t.x;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f*diff*diff : fabsf(diff)-0.5f;
        
        diff = p.y - t.y;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f*diff*diff : fabsf(diff)-0.5f;
        
        diff = p.z - t.z;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f*diff*diff : fabsf(diff)-0.5f;
        
        diff = p.w - t.w;
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f*diff*diff : fabsf(diff)-0.5f;
    }

    // Handle remainder elements in chunk
    int scalar_start = chunk_offset + (vec_count * 4);
    int scalar_end = chunk_offset + chunk_size;
    for (int i = scalar_start + idx; i < scalar_end; i += stride) {
        float diff = __ldg(predictions + i) - __ldg(targets + i);
        thread_sum += (fabsf(diff) < 1.0f) ? 0.5f*diff*diff : fabsf(diff)-0.5f;
    }

    // Block reduction
    __shared__ float shared_mem[256];
    shared_mem[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared_mem[0]);
    }
}

torch::Tensor pipelined_smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input shape mismatch");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Inputs must be contiguous");
    
    const int n_elements = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());
    
    // Create CUDA streams for pipelining
    constexpr int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Process data in chunks using multiple streams
    const int chunk_size = (n_elements + num_streams - 1) / num_streams;
    const int block_size = 256;
    const int grid_size = (chunk_size + block_size - 1) / block_size;

    for (int i = 0; i < num_streams; i++) {
        int current_chunk_size = std::min(chunk_size, n_elements - i * chunk_size);
        if (current_chunk_size <= 0) break;

        pipelined_smooth_l1_kernel<<<grid_size, block_size, 0, streams[i]>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            current_chunk_size,
            i * chunk_size
        );
    }

    // Synchronize all streams and cleanup
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    // Normalize the final result
    auto output_cpu = output.cpu();
    output_cpu.div_(n_elements);
    return output_cpu.cuda();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pipelined_smooth_l1_loss_cuda, "Pipelined Smooth L1 Loss (CUDA)");
}