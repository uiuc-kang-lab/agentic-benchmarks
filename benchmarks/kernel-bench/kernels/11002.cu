#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 512;
static const int VEC_SIZE = 4;

__constant__ float const_preds_buffer[32768]; // 32KB for preds (float*)
__constant__ float const_tgts_buffer[32768];  // 32KB for tgts (float*)

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const int64_t num_elements,
    double* __restrict__ sum_out
) {
    __shared__ double shm[BLOCK_SIZE];
    double thread_sum = 0.0;
    
    int idx = blockIdx.x * blockDim.x * VEC_SIZE + threadIdx.x;
    
    // Vectorized load using constant memory cache
    #pragma unroll
    for(int vec_offset=0; vec_offset<VEC_SIZE; ++vec_offset) {
        int load_idx = idx + vec_offset * blockDim.x;
        if(load_idx < num_elements) {
            float p = const_preds_buffer[load_idx];
            float t = const_tgts_buffer[load_idx];
            double diff = static_cast<double>(p) - static_cast<double>(t);
            thread_sum += diff * diff;
        }
    }

    // Warp-level reduction
    for(int offset=16; offset>0; offset/=2)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    
    if(threadIdx.x % 32 == 0)
        shm[threadIdx.x/32] = thread_sum;
    __syncthreads();

    // Final block reduction
    if(threadIdx.x == 0) {
        double block_sum = 0.0;
        for(int i=0; i<BLOCK_SIZE/32; i++)
            block_sum += shm[i];
        atomicAdd(sum_out, block_sum);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(),
               "Inputs must be CUDA tensors");
    TORCH_CHECK(predictions.sizes() == targets.sizes(),
               "Input sizes must match");

    const int64_t num_elements = predictions.numel();
    constexpr int64_t const_max = sizeof(const_preds_buffer)/sizeof(float);
    
    TORCH_CHECK(num_elements <= const_max,
               "Input size exceeds constant memory capacity");

    // Copy input data to constant memory
    cudaMemcpyToSymbolAsync(const_preds_buffer,
                           predictions.data_ptr<float>(),
                           num_elements*sizeof(float));
    cudaMemcpyToSymbolAsync(const_tgts_buffer,
                           targets.data_ptr<float>(),
                           num_elements*sizeof(float));

    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));
    
    const int grid_size = (num_elements + BLOCK_SIZE*VEC_SIZE - 1) / (BLOCK_SIZE*VEC_SIZE);

    mse_forward_kernel<float><<<grid_size, BLOCK_SIZE>>>(
        num_elements,
        accumulator.data_ptr<double>()
    );

    return accumulator.div_(num_elements).to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE forward with constant mem & vectorization");
}