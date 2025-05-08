#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel processes the bulk of the input using vectorized float4 loads/stores,
// and then handles the remaining tail elements using warp-level primitives
// (i.e., __ballot_sync and __shfl_down_sync) to determine, in a warp-synchronous manner,
// how many elements remain. This avoids using shared memory for such a small, specialized task.

__global__ void tanh_warp_shuffle_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           const int numel) {
  int total_threads = blockDim.x * gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Process full groups of 4 floats using vectorized loads
  int numVec = numel / 4;
  for (int i = tid; i < numVec; i += total_threads) {
    float4 in = reinterpret_cast<const float4*>(input)[i];
    float4 out;
    out.x = tanhf(in.x);
    out.y = tanhf(in.y);
    out.z = tanhf(in.z);
    out.w = tanhf(in.w);
    reinterpret_cast<float4*>(output)[i] = out;
  }

  // Process tail elements using warp-level primitives
  int tail_start = numVec * 4;
  int tail_count = numel - tail_start;  // remainder in [0, 3]

  // Let the first warp of block 0 handle the tail elements
  if (blockIdx.x == 0 && threadIdx.x < 32) {
    // Each lane in the warp checks if its lane index is within the tail count
    int flag = (threadIdx.x < tail_count) ? 1 : 0;
    // Perform a warp-level reduction using __shfl_down_sync to sum the flags
    for (int offset = 16; offset > 0; offset /= 2) {
      flag += __shfl_down_sync(0xffffffff, flag, offset);
    }
    // Lane 0 now has the total number of tail elements
    if (threadIdx.x == 0) {
      for (int i = 0; i < flag; i++) {
        int idx = tail_start + i;
        if (idx < numel) {
          output[idx] = tanhf(input[idx]);
        }
      }
    }
  }
}

// Fallback generic kernel for non-float types using a grid-stride loop
template <typename scalar_t>
__global__ void tanh_generic_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      int numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < numel; i += stride) {
    output[i] = tanh(input[i]);
  }
}

// Forward function exposed to Python
torch::Tensor forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  int numel = input.numel();
  const int threads = 256;

  if (input.scalar_type() == at::ScalarType::Float) {
    int numVec = numel / 4;
    int blocks = (numVec + threads - 1) / threads;
    blocks = (blocks > 0) ? blocks : 1;
    tanh_warp_shuffle_kernel<<<blocks, threads>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      numel
    );
  } else {
    int blocks = (numel + threads - 1) / threads;
    blocks = (blocks > 0) ? blocks : 1;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_generic_kernel", ([&] {
      tanh_generic_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        numel
      );
    }));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Tanh forward with warp-level tail processing (CUDA)");
}
