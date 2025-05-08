#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS = 256;
const int ELEMENTS_PER_THREAD = 4;
const int SHARED_MEM_SIZE = THREADS * ELEMENTS_PER_THREAD;

// Kernel optimized by unrolling loops

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                             scalar_t* __restrict__ output,
                             const int64_t size) {
    __shared__ float shared_data[SHARED_MEM_SIZE];
    
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * SHARED_MEM_SIZE;
    
    using Vec4T = float4;
    const Vec4T* input_vec = reinterpret_cast<const Vec4T*>(input + block_offset);
    Vec4T* output_vec = reinterpret_cast<Vec4T*>(output + block_offset);
    
    if (block_offset + tid * 4 + 3 < size) {
        Vec4T in_vec = input_vec[tid];
        shared_data[tid * 4] = in_vec.x;
        shared_data[tid * 4 + 1] = in_vec.y;
        shared_data[tid * 4 + 2] = in_vec.z;
        shared_data[tid * 4 + 3] = in_vec.w;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = block_offset + tid * 4 + i;
            if (idx < size) {
                shared_data[tid * 4 + i] = static_cast<float>(input[idx]);
            }
        }
    }
    // Synchronize only after loading data into shared memory
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int idx = block_offset + tid * 4 + i;
        if (idx < size) {
            float val = -shared_data[tid * 4 + i];
            float exp_val = __expf(val);
            float r = __fdividef(1.0f, (1.0f + exp_val));
            shared_data[tid * 4 + i] = r;
        }
    }
    // Synchronize only if data is needed by other threads
    __syncthreads();
    
    if (block_offset + tid * 4 + 3 < size) {
        Vec4T out_vec;
        out_vec.x = shared_data[tid * 4];
        out_vec.y = shared_data[tid * 4 + 1];
        out_vec.z = shared_data[tid * 4 + 2];
        out_vec.w = shared_data[tid * 4 + 3];
        output_vec[tid] = out_vec;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = block_offset + tid * 4 + i;
            if (idx < size) {
                output[idx] = static_cast<scalar_t>(shared_data[tid * 4 + i]);
            }
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    const int blocks = (size + SHARED_MEM_SIZE - 1) / SHARED_MEM_SIZE;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        
        sigmoid_kernel<scalar_t><<<blocks, THREADS>>>(input_data, output_data, size);
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Sigmoid forward (CUDA)");
}
