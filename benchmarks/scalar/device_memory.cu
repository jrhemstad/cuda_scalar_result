
#include "kernels.cuh"

#include <benchmark/benchmark.h>

template <typename T> static void BM_device_memory(::benchmark::State &state) {
  auto size = state.range(0);
  T *d_result{};
  T h_result{};
  cudaMalloc(&d_result, sizeof(T));
  constexpr std::size_t block_size{256};
  auto grid_size = (size + block_size + 1) / size;

  for (auto _ : state) {
    cudaMemset(d_result, 0, sizeof(T));
    simple_reduction<block_size><<<block_size, grid_size>>>(size, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(h_result), cudaMemcpyDefault);
  }
  cudaFree(d_result);
}
BENCHMARK_TEMPLATE(BM_device_memory, int)
    ->Apply(generate_size)
    ->Unit(benchmark::kMicrosecond);