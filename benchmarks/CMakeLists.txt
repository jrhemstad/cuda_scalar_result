cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(EXAMPLE LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER $ENV{tCC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

CPMAddPackage("gh:NVIDIA/libcudacxx#d4bd0f6de28b265f7428e39340cbe28c1b560da6")

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    add_executable(${CMAKE_BENCH_NAME}
                   ${CMAKE_BENCH_SRC})
    #set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_BENCH_NAME} libcudacxx::libcudacxx benchmark benchmark_main pthread )
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
    target_compile_features(${CMAKE_BENCH_NAME} PRIVATE cxx_std_17 cuda_std_17)
endfunction(ConfigureBench)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/cub"
                    "${GBENCH_INCLUDE_DIR}"
                    "${CMAKE_CURRENT_SOURCE_DIR}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${GBENCH_LIBRARY_DIR}")

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - example benchmark --------------------------------------------------------------------------------

set(EXAMPLE_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/example/example_benchmark.cpp")

ConfigureBench(EXAMPLE_BENCH "${EXAMPLE_BENCH_SRC}")

# - example benchmark --------------------------------------------------------------------------------

ConfigureBench(DEVICE_MEMORY "${CMAKE_CURRENT_SOURCE_DIR}/scalar/device_memory.cu")

ConfigureBench(MANAGED_MEMORY "${CMAKE_CURRENT_SOURCE_DIR}/scalar/managed_memory.cu")

ConfigureBench(MANAGED_MEMORY_PREFETCH "${CMAKE_CURRENT_SOURCE_DIR}/scalar/managed_memory_prefetch.cu")

ConfigureBench(PINNED_MEMORY "${CMAKE_CURRENT_SOURCE_DIR}/scalar/pinned_memory.cu")

ConfigureBench(CLEVER_PINNED "${CMAKE_CURRENT_SOURCE_DIR}/scalar/clever_pinned.cu")

ConfigureBench(ATOMIC_PINNED "${CMAKE_CURRENT_SOURCE_DIR}/scalar/atomic_pinned.cu")

###################################################################################################