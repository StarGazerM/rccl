cmake_minimum_required(VERSION 3.26)
project(rccl_memory_rmm_parity LANGUAGES CXX)

find_package(rmm CONFIG REQUIRED)

add_executable(rmm_allocation_trace rmm_allocation_trace.cpp)
target_compile_features(rmm_allocation_trace PRIVATE cxx_std_20)
target_link_libraries(rmm_allocation_trace PRIVATE rmm::rmm)
