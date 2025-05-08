#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel using grid-stride loop for even workload distribution
template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int total_elements, int M, int L, int K) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Loop over all output elements in a grid-stride fashion
    for (int index = idx; index < total_elements; index += stride) {
        // Compute 3D indices from the flattened index
        int n = index / (M * L);
        int rem = index % (M * L);
        int m = rem / L;
        int l = rem % L;

        scalar_t sum = 0;
        int k = 0;
        int a_base = n * M * K + m * K;
        // Unroll loop by factor of 4 with fast math for float types
        #pragma unroll
        for (; k <= K - 4; k += 4) {
            if constexpr (std::is_same<scalar_t, float>::value) {
                sum = __fmaf_rn(A[a_base + k    ], B[(k    ) * L + l], sum);
                sum = __fmaf_rn(A[a_base + k + 1], B[(k + 1) * L + l], sum);
                sum = __fmaf_rn(A[a_base + k + 2], B[(k + 2) * L + l], sum);
                sum = __fmaf_rn(A[a_base + k + 3], B[(k + 3) * L + l], sum);
            } else {
                sum += A[a_base + k    ] * B[(k    ) * L + l];
                sum += A[a_base + k + 1] * B[(k + 1) * L + l];
                sum += A[a_base + k + 2] * B[(k + 2) * L + l];
                sum += A[a_base + k + 3] * B[(k + 3) * L + l];
            }
        }
        for (; k < K; k++) {
            if constexpr (std::is_same<scalar_t, float>::value) {
                sum = __fmaf_rn(A[a_base + k], B[k * L + l], sum);
            } else {
                sum += A[a_base + k] * B[k * L + l];
            }
        }
        output[index] = sum;
    }
}

// CUDA forward function
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    int total_elements = N * M * L;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        module_fn_cuda_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_elements, M, L, K);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)  CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
  CHECK_INPUT(A);
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
