
#include <cub/cub.cuh>

template <std::size_t block_size, typename T>
__global__ void simple_reduction(std::size_t input_size, T *global_result) {

  auto tid = threadIdx.x + blockIdx.x * gridDim.x;

  T thread_data{};

  while (tid < input_size) {
    ++thread_data;
    tid += blockDim.x * gridDim.x;
  }

  using BlockReduce = cub::BlockReduce<T, block_size>;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  T block_result = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0)
    atomicAdd(global_result, block_result);
}
