#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L,
    int start_n, int end_n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_elements = (end_n - start_n) * M * L;

    if (idx < chunk_elements) {
        int n = start_n + idx / (M * L);
        int m = (idx % (M * L)) / L;
        int l = idx % L;

        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[n * M * K + m * K + k] * B[k * L + l];
        }
        output[n * M * L + m * L + l] = sum;
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int chunk_size = (N + num_streams - 1) / num_streams;
    const int threads = 1024;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        for (int i = 0; i < num_streams; ++i) {
            int start_n = i * chunk_size;
            int end_n = min((i+1)*chunk_size, N);
            int elements = (end_n - start_n) * M * L;
            if (elements == 0) continue;

            int blocks = (elements + threads - 1) / threads;
            module_fn_cuda_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N, M, K, L,
                start_n, end_n);
        }
    }));

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
  TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
  TORCH_CHECK(A.dim() == 3, "Input A must be a 3D tensor");
  TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
  TORCH_CHECK(B.dim() == 2, "Input B must be a 2D tensor");
  CHECK_INPUT(B);

  auto N = A.size(0);
  auto M = A.size(1);
  auto L = B.size(1);

  auto output = torch::zeros({N, M, L}, A.options());
  module_fn_cuda_forward(A, B, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "module_fn forward (CUDA)");
}
