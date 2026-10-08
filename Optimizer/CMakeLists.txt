cmake_minimum_required(VERSION 3.12)
project(PINN_CPP_Runner)

set(CMAKE_CXX_STANDARD 17)

# Find Python 3 headers and development libraries
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)

# Define executable
add_executable(pinn_cpp main.cpp)

# Link Python include directories and libraries
target_include_directories(pinn_cpp PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(pinn_cpp PRIVATE ${Python3_LIBRARIES})
