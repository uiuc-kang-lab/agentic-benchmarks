#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cumprod_hybrid_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {
    
    extern __shared__ scalar_t shared_products[];
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_batches;
         idx += blockDim.x * gridDim.x) {
        
        const int batch_idx = idx / stride;
        const int in_idx = idx % stride;
        const int64_t base_idx = batch_idx * (dim_size * stride) + in_idx;
        
        scalar_t product = 1;
        int i = 0;
        
        #pragma unroll 16
        for (; i + 15 < dim_size; i += 16) {
            const int64_t idx0 = base_idx + i * stride;
            const int64_t idx8 = base_idx + (i + 8) * stride;
            
            scalar_t val0 = input[idx0];
            scalar_t val1 = input[idx0 + stride];
            scalar_t val2 = input[idx0 + 2 * stride];
            scalar_t val3 = input[idx0 + 3 * stride];
            scalar_t val4 = input[idx0 + 4 * stride];
            scalar_t val5 = input[idx0 + 5 * stride];
            scalar_t val6 = input[idx0 + 6 * stride];
            scalar_t val7 = input[idx0 + 7 * stride];
            
            product *= val0;
            output[idx0] = product;
            product *= val1;
            output[idx0 + stride] = product;
            product *= val2;
            output[idx0 + 2 * stride] = product;
            product *= val3;
            output[idx0 + 3 * stride] = product;
            product *= val4;
            output[idx0 + 4 * stride] = product;
            product *= val5;
            output[idx0 + 5 * stride] = product;
            product *= val6;
            output[idx0 + 6 * stride] = product;
            product *= val7;
            output[idx0 + 7 * stride] = product;
            
            val0 = input[idx8];
            val1 = input[idx8 + stride];
            val2 = input[idx8 + 2 * stride];
            val3 = input[idx8 + 3 * stride];
            val4 = input[idx8 + 4 * stride];
            val5 = input[idx8 + 5 * stride];
            val6 = input[idx8 + 6 * stride];
            val7 = input[idx8 + 7 * stride];
            
            product *= val0;
            output[idx8] = product;
            product *= val1;
            output[idx8 + stride] = product;
            product *= val2;
            output[idx8 + 2 * stride] = product;
            product *= val3;
            output[idx8 + 3 * stride] = product;
            product *= val4;
            output[idx8 + 4 * stride] = product;
            product *= val5;
            output[idx8 + 5 * stride] = product;
            product *= val6;
            output[idx8 + 6 * stride] = product;
            product *= val7;
            output[idx8 + 7 * stride] = product;
        }
        
        for (; i < dim_size; i++) {
            const int64_t curr_idx = base_idx + i * stride;
            product *= input[curr_idx];
            output[curr_idx] = product;
        }
    }
}

torch::Tensor cumprod_hybrid_forward(torch::Tensor input, int64_t dim) {
    bool input_on_cpu = !input.is_cuda();
    torch::Tensor input_device = input_on_cpu ? 
        input.to(torch::kCUDA, /*non_blocking=*/true) : input;
    
    auto output_device = torch::empty_like(input_device);
    
    cudaStream_t compute_stream;
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
    
    const auto sizes = input_device.sizes();
    const auto strides = input_device.strides();
    const int64_t dim_size = sizes[dim];
    const int64_t stride = strides[dim];
    const int64_t total_batches = input_device.numel() / dim_size;
    
    const int threads = 256;
    const int blocks = std::min(65535, (int)((total_batches + threads - 1) / threads));
    
    const size_t shared_mem_size = threads * sizeof(typename std::iterator_traits<scalar_t>::value_type);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_device.scalar_type(), "cumprod_hybrid", ([&] {
        cumprod_hybrid_kernel<scalar_t><<<blocks, threads, shared_mem_size, compute_stream>>>(
            output_device.data_ptr<scalar_t>(),
            input_device.data_ptr<scalar_t>(),
            dim_size,
            stride,
            total_batches
        );
    }));
    
    torch::Tensor output;
    if (input_on_cpu) {
        output = torch::empty_like(input);
        cudaMemcpyAsync(
            output.data_ptr(),
            output_device.data_ptr(),
            output_device.numel() * output_device.element_size(),
            cudaMemcpyDeviceToHost,
            compute_stream
        );
    } else {
        output = output_device;
    }
    
    cudaStreamSynchronize(compute_stream);
    cudaStreamDestroy(compute_stream);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_hybrid_forward, "Hybrid optimized cumulative product forward (CUDA)");
}