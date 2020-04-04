cmake_minimum_required(VERSION 3.15)
project(cuda)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)


#CUDA_ADD_EXECUTABLE(cuda cuda_inpainting.cu)
cuda_add_executable(cuda
        cuda_inpainting.cu
        cuda_inpainting.h
        inpainting.cpp
        Patch.cpp
        Patch.h)

target_link_libraries(cuda ${OpenCV_LIBS})
