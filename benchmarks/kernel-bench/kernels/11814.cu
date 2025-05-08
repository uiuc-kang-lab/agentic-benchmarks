#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
__device__ inline float4 load_float4(const T* base, int idx) {
    return reinterpret_cast<const float4*>(&base[idx])[0];
}

__global__ void kl_div_kernel_shared(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {
    
    extern __shared__ float shared_data[];
    const int elements_per_thread = 4;
    const int tile_size = blockDim.x * elements_per_thread;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    float thread_sum = 0.0f;
    
    for (int base_idx = bid * tile_size; base_idx < n; base_idx += tile_size * gridDim.x) {
        int load_idx = base_idx + tid * elements_per_thread;
        int valid_load_size = min(elements_per_thread, n - base_idx - tid * elements_per_thread);
        
        // Load to shared memory using float4 for coalesced access
        if (load_idx + 3 < n) {
            float4 log_pred4 = load_float4(log_predictions, load_idx);
            float4 targets4 = load_float4(targets, load_idx);
            
            #pragma unroll
            for (int i = 0; i < elements_per_thread; ++i) {
                shared_data[tid * elements_per_thread + i] = ((float*)&log_pred4)[i];
                shared_data[tile_size + tid * elements_per_thread + i] = ((float*)&targets4)[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < elements_per_thread; ++i) {
                if (load_idx + i < n) {
                    shared_data[tid * elements_per_thread + i] = log_predictions[load_idx + i];
                    shared_data[tile_size + tid * elements_per_thread + i] = targets[load_idx + i];
                }
            }
        }
        __syncthreads();
        
        // Process tile from shared memory
        int process_limit = min(tile_size, n - base_idx);
        for (int i = 0; i < process_limit; i++) {
            float log_pred = shared_data[i];
            float target = shared_data[tile_size + i];
            thread_sum += __expf(log_pred) - target * log_pred;
        }
        __syncthreads();
    }

    // Warp-reduce then atomic add
    for (int offset = 16; offset > 0; offset /= 2)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    
    if ((tid % 32) == 0)
        atomicAdd(output, thread_sum);
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int elements_per_thread = 4;
    const int shared_mem = threads * elements_per_thread * 2 * sizeof(float);
    
    // Calculate maximum blocks to fit in SM resources
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    int max_blocks = prop.multiProcessorCount * 16;
    int blocks = min((n + threads * elements_per_thread - 1) / (threads * elements_per_thread), max_blocks);
    
    kl_div_kernel_shared<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with shared tiling (CUDA)");
}