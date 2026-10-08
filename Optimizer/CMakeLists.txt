cmake_minimum_required(VERSION 3.12)
project(PINN_CPP_Runner)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
add_executable(pinn_cpp main.cpp)

target_include_directories(pinn_cpp PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(pinn_cpp PRIVATE ${Python3_LIBRARIES})
