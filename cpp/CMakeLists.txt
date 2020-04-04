cmake_minimum_required(VERSION 3.15)
project(cpp)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
include_directories(.)

add_executable(cpp
        inpainting.cpp
        Makefile
        Patch.cpp
        Patch.h)

target_link_libraries(cpp ${OpenCV_LIBS})