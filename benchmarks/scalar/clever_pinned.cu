

#include "kernels.cuh"

#include <benchmark/benchmark.h>
#include <cuda/std/atomic>
#include <new>

template <std::size_t block_size>
__global__ void clever_reduction(
    std::size_t input_size,
    int *final_result,
    int *temp_accumulator,
    int *count)
{
   auto tid = threadIdx.x + blockIdx.x * gridDim.x;
   int thread_data{};
   // Assume launching enough blocks to cover the entire input
   if (tid < input_size){
      ++thread_data;
   }
   using BlockReduce = cub::BlockReduce<int, block_size>;
   __shared__ typename BlockReduce::TempStorage temp_storage;
   auto const block_result = BlockReduce(temp_storage).Sum(thread_data);

   if (threadIdx.x == 0){
      atomicAdd(temp_accumulator, block_result);
      auto const is_last_block = (gridDim.x - 1) == atomicAdd(count, 1);
      if (is_last_block){
         // Store final reduction to pinned memory
         *final_result = *temp_accumulator;
      }
   }
}

static void BM_clever_pinned(::benchmark::State &state)
{
   int *storage;
   cudaMalloc(&storage, 2 * sizeof(int));

   int *d_temp_accumulator = &storage[0];
   int *d_count = &storage[1];

   int *hd_result;
   cudaMallocHost(&hd_result, sizeof(int));

   constexpr std::size_t block_size{256};
   auto size = state.range(0);
   auto grid_size = (size + block_size + 1) / size;

   int h_result;

   for (auto _ : state)
   {
      // Zero storage
      cudaMemset(storage, 0, 2 * sizeof(int));
      *hd_result = 0;

      clever_reduction<block_size><<<block_size, grid_size>>>(size, hd_result, d_temp_accumulator, d_count);

      // Spin waiting for result from kernel
      while (hd_result == 0);

      benchmark::DoNotOptimize(h_result = *hd_result);
   }
   cudaFreeHost(hd_result);
   cudaFree(storage);
}

BENCHMARK(BM_clever_pinned)
    ->Apply(generate_size)
    ->Unit(benchmark::kMicrosecond);
