

#include "kernels.cuh"

#include <benchmark/benchmark.h>
#include <cuda/std/atomic>
#include <iostream>
#include <new>
#include <thrust/iterator/iterator_traits.h>

template <typename T> __global__ void zero(T *count) { new (count) T{0}; }

template <std::size_t block_size, typename T>
__global__ void std_reduction(
    std::size_t input_size,
    cuda::atomic<T, cuda::thread_scope_system> *global_result,
    cuda::atomic<T, cuda::thread_scope_device> *device_result,
    cuda::atomic<unsigned int, cuda::thread_scope_device> *atomic_count) {
  auto tid = threadIdx.x + blockIdx.x * gridDim.x;
  T thread_data{};
  while (tid < input_size) {
    ++thread_data;
    tid += blockDim.x * gridDim.x;
  }
  using BlockReduce = cub::BlockReduce<T, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T block_result = BlockReduce(temp_storage).Sum(thread_data);
  bool is_last_block_done = false;
  if (threadIdx.x == 0) {
    device_result->fetch_add(block_result, cuda::std::memory_order_relaxed);
    unsigned value = atomic_count->fetch_add(1, cuda::memory_order_release);
    is_last_block_done = value == (gridDim.x - 1);
  }
  if (is_last_block_done) {
    // copy result to global buffer
    if (threadIdx.x == 0) {
      global_result->store(device_result->load(cuda::std::memory_order_relaxed),
                           cuda::std::memory_order_relaxed);
      device_result->store(
          0, cuda::std::memory_order_relaxed); // set to zero for next time
      atomic_count->store(
          0, cuda::std::memory_order_relaxed); // set to zero for next time
    }
  }
}


template <typename T>
static void BM_std_pinned_memory(::benchmark::State &state) {

  auto size = state.range(0);
  using count_t = cuda::atomic<unsigned int, cuda::thread_scope_device>;
  count_t *atomic_count;
  cudaMalloc(&atomic_count, sizeof(count_t));
  zero<<<1, 1>>>(atomic_count);
  using device_atomic = cuda::atomic<T, cuda::thread_scope_device>;
  device_atomic *d_result{};
  cudaMalloc(&d_result, sizeof(device_atomic));
  zero<<<1, 1>>>(d_result);
  using system_atomic = cuda::atomic<T, cuda::thread_scope_system>;
  system_atomic *hd_result{};
  cudaMallocHost(&hd_result, sizeof(system_atomic));
  T h_result{};
  constexpr std::size_t block_size{256};
  auto grid_size = (size + block_size + 1) / size;
  for (auto _ : state) {
    *hd_result = 0;
    std_reduction<block_size>
        <<<block_size, grid_size>>>(size, hd_result, d_result, atomic_count);
    while (!hd_result->load(cuda::memory_order_acquire))
      ;
    benchmark::DoNotOptimize(h_result = *hd_result);
  }
  cudaFreeHost(hd_result);
  cudaFree(d_result);
  cudaFree(atomic_count);
}
BENCHMARK_TEMPLATE(BM_std_pinned_memory, int)
    ->Apply(generate_size)
    ->Unit(benchmark::kMicrosecond);



