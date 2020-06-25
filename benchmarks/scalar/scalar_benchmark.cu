
#include <benchmark/benchmark.h>
#include <iostream>
#include <cub/cub.cuh> 
#include <thrust/iterator/iterator_traits.h>

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

template <typename T> static void BM_device_memory(::benchmark::State &state) {

  auto size = 1'000'000;

  T *d_result{};

  cudaMalloc(&d_result, sizeof(T));

  constexpr std::size_t block_size{256};

  auto grid_size = (size + block_size + 1) / size;

  for (auto _ : state) {
    cudaMemset(d_result, 0, sizeof(T));
    simple_reduction<block_size><<<block_size, grid_size>>>(size, d_result);
    cudaDeviceSynchronize();
  }
}
BENCHMARK_TEMPLATE(BM_device_memory, int)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 18);
