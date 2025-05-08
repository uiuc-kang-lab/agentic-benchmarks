#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_vectorized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Use dynamically-allocated shared memory with padding to avoid bank conflicts
    extern __shared__ float shmem[];  // each thread gets 5 floats (4 values + 1 padding)
    const int tid = threadIdx.x;
    const int vector_index = blockIdx.x * blockDim.x + tid;
    const int idx = 4 * vector_index;
    const int sh_offset = tid * 5;  // padding: 5 floats per thread

    // Load input into shared memory
    if (idx < size) {
        float4 val;
        if (idx + 3 < size) {
            val = reinterpret_cast<const float4*>(input)[vector_index];
        } else {
            val.x = (idx < size) ? input[idx] : 0;
            val.y = (idx + 1 < size) ? input[idx + 1] : 0;
            val.z = (idx + 2 < size) ? input[idx + 2] : 0;
            val.w = (idx + 3 < size) ? input[idx + 3] : 0;
        }
        shmem[sh_offset    ] = val.x;
        shmem[sh_offset + 1] = val.y;
        shmem[sh_offset + 2] = val.z;
        shmem[sh_offset + 3] = val.w;
    }
    
    __syncthreads();
    
    // Compute tanh on data in shared memory
    if (idx < size) {
        float a = tanhf(shmem[sh_offset    ]);
        float b = tanhf(shmem[sh_offset + 1]);
        float c = tanhf(shmem[sh_offset + 2]);
        float d = tanhf(shmem[sh_offset + 3]);
        shmem[sh_offset    ] = a;
        shmem[sh_offset + 1] = b;
        shmem[sh_offset + 2] = c;
        shmem[sh_offset + 3] = d;
    }
    
    __syncthreads();
    
    // Write results from shared memory back to global memory
    if (idx < size) {
        if (idx + 3 < size) {
            float4 res;
            res.x = shmem[sh_offset    ];
            res.y = shmem[sh_offset + 1];
            res.z = shmem[sh_offset + 2];
            res.w = shmem[sh_offset + 3];
            reinterpret_cast<float4*>(output)[vector_index] = res;
        } else {
            for (int i = 0; i < 4 && idx + i < size; ++i) {
                output[idx + i] = shmem[sh_offset + i];
            }
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 512;
    const int elements_per_block = threads * 4;
    const int blocks = (input.numel() + elements_per_block - 1) / elements_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_vectorized_kernel", ([&] {
        tanh_vectorized_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized Tanh optimized for H100 (CUDA)");
}