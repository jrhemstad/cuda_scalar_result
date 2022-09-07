#include "kernels.cuh"

#include <benchmark/benchmark.h>

template <typename T>
static void BM_managed_memory_prefetch(::benchmark::State &state) {
  auto size = state.range(0);
  T *d_result{};
  T h_result{};
  cudaMallocManaged(&d_result, sizeof(T));
  constexpr std::size_t block_size{256};
  auto grid_size = (size + block_size + 1) / size;

  for (auto _ : state) {
    *d_result = 0;
    cudaMemPrefetchAsync(d_result, sizeof(T), 0);
    simple_reduction<block_size><<<block_size, grid_size>>>(size, d_result);
    cudaDeviceSynchronize();
    benchmark::DoNotOptimize(h_result = *d_result);
  }
  cudaFree(d_result);
}
BENCHMARK_TEMPLATE(BM_managed_memory_prefetch, int)
    ->Apply(generate_size)
    ->Unit(benchmark::kMicrosecond);
