

#include "kernels.cuh"

#include <benchmark/benchmark.h>
#include <cuda/atomic>
#include <new>

template <std::size_t block_size>
__global__ void std_reduction(
    std::size_t input_size,
    int *final_result,
    int *temp_accumulator,
    int *count)
{
   auto tid = threadIdx.x + blockIdx.x * gridDim.x;
   int thread_data{};
   while (tid < input_size)
   {
      ++thread_data;
      tid += blockDim.x * gridDim.x;
   }
   using BlockReduce = cub::BlockReduce<int, block_size>;
   __shared__ typename BlockReduce::TempStorage temp_storage;
   int block_result = BlockReduce(temp_storage).Sum(thread_data);

   cuda::atomic_ref<int, cuda::thread_scope_device> temp_accumulator_ref{*temp_accumulator};
   cuda::atomic_ref<int, cuda::thread_scope_device> count_ref{*count};
   if (threadIdx.x == 0)
   {
      temp_accumulator_ref.fetch_add(block_result, cuda::memory_order_relaxed);

      // Use acq_rel to guarantee temp_accumulator_ref.fetch_add is not reordered after incrementing the count
      // and ensure that other thread's updates to temp_accumulator_ref are visible
      auto const is_last_block = (count_ref.fetch_add(1, cuda::memory_order_acq_rel) == (gridDim.x - 1));
      if (is_last_block)
      {
         // Guaranteed that all other thread updates are visible
         auto const final_sum = temp_accumulator_ref.load(cuda::memory_order_relaxed); 
         cuda::atomic_ref<int, cuda::thread_scope_system>{*final_result}.store(final_sum, cuda::memory_order_relaxed);
         
         temp_accumulator_ref.store(0, cuda::std::memory_order_relaxed); // set to zero for next timek
         count_ref.store(0, cuda::std::memory_order_relaxed);  // set to zero for next time
      }
   }
}

template <typename T>
static void BM_std_pinned_memory(::benchmark::State &state)
{
   auto size = state.range(0);

   int *storage;
   cudaMalloc(&storage, 2 * sizeof(int));

   int *hd_result{};
   cudaMallocHost(&hd_result, sizeof(int));
   T h_result{};
   constexpr std::size_t block_size{256};
   auto grid_size = (size + block_size + 1) / size;

   for (auto _ : state)
   {
      *hd_result = 0;
      std_reduction<block_size><<<block_size, grid_size>>>(size, hd_result, &storage[0], &storage[1]);
      cuda::atomic_ref<int, cuda::thread_scope_system> hd_result_ref{*hd_result};
      while (hd_result_ref.load(cuda::memory_order_relaxed) == 0);
      benchmark::DoNotOptimize(h_result = *hd_result);
   }

   cudaFreeHost(hd_result);
   cudaFree(storage);
}
BENCHMARK_TEMPLATE(BM_std_pinned_memory, int)
    ->Apply(generate_size)
    ->Unit(benchmark::kMicrosecond);
