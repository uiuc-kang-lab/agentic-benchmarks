#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void l2_norm_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {

    const int vec_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int base_offset = vec_idx * outer_stride;

    scalar_t sum = 0.0;
    for (int i = tid; i < C; i += blockDim.x) {
        scalar_t val = input[base_offset + i * stride_C];
        sum += val * val;
    }

    __shared__ scalar_t shared[256];
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        norms[vec_idx] = shared[0];
    }
}

template <typename scalar_t>
__global__ void l2_norm_normalize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {

    const int vec_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int base_offset = vec_idx * outer_stride;

    scalar_t norm_val = sqrt(norms[vec_idx]) + 1e-12;
    scalar_t inv_norm = 1.0 / norm_val;

    for (int i = tid; i < C; i += blockDim.x) {
        output[base_offset + i * stride_C] = input[base_offset + i * stride_C] * inv_norm;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have >=2 dimensions");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);
    auto norms = torch::empty({total_vectors}, input.options());

    const int threads = 256;
    const int split = (total_vectors + 1) / 2;
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_reduce", [&] {
        l2_norm_reduce_kernel<scalar_t><<<split, threads, 0, stream1.stream()>>>(
            input.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
        
        l2_norm_reduce_kernel<scalar_t><<<total_vectors - split, threads, 0, stream2.stream()>>>(
            input.data_ptr<scalar_t>() + split * outer_stride,
            norms.data_ptr<scalar_t>() + split,
            C,
            stride_C,
            outer_stride
        );
    });

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_normalize", [&] {
        l2_norm_normalize_kernel<scalar_t><<<split, threads, 0, stream1.stream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
        
        l2_norm_normalize_kernel<scalar_t><<<total_vectors - split, threads, 0, stream2.stream()>>>(
            input.data_ptr<scalar_t>() + split * outer_stride,
            output.data_ptr<scalar_t>() + split * outer_stride,
            norms.data_ptr<scalar_t>() + split,
            C,
            stride_C,
            outer_stride
        );
    });

    cudaStreamSynchronize(stream1.stream());
    cudaStreamSynchronize(stream2.stream());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with stream overlapping");
}
