#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_BLOCKS 65535

template<typename T>
__device__ __forceinline__ T warp_reduce_prod(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void prod_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim_size,
    int stride,
    int num_elements,
    int elements_per_block
) {
    extern __shared__ float sdata[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * elements_per_block;
    const int lane = tid % WARP_SIZE;
    const int wid = tid / WARP_SIZE;
    
    for (int elem = 0; elem < elements_per_block && (gid + elem) < num_elements; elem++) {
        const int outIdx = gid + elem;
        float partial = 1.0f;
        
        for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
            partial *= input[outIdx + i * stride];
        }
        
        partial = warp_reduce_prod(partial);
        
        if (lane == 0) {
            sdata[wid + elem * (BLOCK_SIZE/WARP_SIZE)] = partial;
        }
        __syncthreads();
        
        if (tid < (BLOCK_SIZE/WARP_SIZE)) {
            partial = (tid < (BLOCK_SIZE/WARP_SIZE)) ? sdata[tid + elem * (BLOCK_SIZE/WARP_SIZE)] : 1.0f;
            partial = warp_reduce_prod(partial);
            
            if (tid == 0) {
                output[outIdx] = partial;
            }
        }
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());
    
    int num_elements = output.numel();
    int stride = x.stride(dim);
    
    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    int elements_per_block = 4;
    int num_blocks = (num_elements + elements_per_block - 1) / elements_per_block;
    num_blocks = min(num_blocks, MAX_BLOCKS);
    
    size_t shared_mem_size = (BLOCK_SIZE/WARP_SIZE) * elements_per_block * sizeof(float);
    
    prod_reduction_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        input_ptr, output_ptr, dim_size, stride, num_elements, elements_per_block
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}