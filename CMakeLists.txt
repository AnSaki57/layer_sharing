cmake_minimum_required(VERSION 3.10)
project(FedFault)

set(CMAKE_CXX_STANDARD 17)

# Point to LibTorch local path
set(CMAKE_PREFIX_PATH "/home/ansaki57/Documents/ClassesFiles/Sem6/MiniProject/libtorch-cxx11-abi-shared-with-deps-1.10.0+cpu/libtorch")
find_package(Torch REQUIRED)

add_executable(fedFaultRandLayers fedFaultRandLayers.cpp)
target_link_libraries(fedFaultRandLayers "${TORCH_LIBRARIES}")
set_property(TARGET fedFaultRandLayers PROPERTY CXX_STANDARD 17)

find_package(Threads REQUIRED)
target_link_libraries(fedFaultRandLayers Threads::Threads)
